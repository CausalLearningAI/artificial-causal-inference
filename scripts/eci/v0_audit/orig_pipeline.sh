#!/bin/bash
#
# Run the ORIGINAL ECI repo's own scripts (tmencatt/projects/neuralscience, read-only; a copy is made in the
# job's stage dir) on our ants v2 data: manifest (orig_prepare.py) -> their s2-encode_dataset (DINOV2) ->
# their s3-train_sae (TOPK, k 20, 10 epochs, batch $BATCH, unseeded as in the original) -> their
# s4-sae_encoding with --pooler MEAN and --pooler CLS (CLS = token 0 = top-left patch; the old default) ->
# orig_analyse.py (notebook NES + pre-registered alignment). CPU only (GPU queue ~10 days on 2026-10-20).
#   BATCH=128 TAG=b128_r1 sbatch --export=ALL scripts/eci/v0_audit/orig_pipeline.sh
# The only edits to the copy: s3's hard-coded SAVING_PATH -> stage dir; a 3-line jaxtyping stub (type hints
# only) because the crl env lacks jaxtyping; wandb disabled via WANDB_MODE.
#
#SBATCH --job-name=eci_v0_orig
#SBATCH --output=logs/eci_v0_orig_%j.out
#SBATCH --error=logs/eci_v0_orig_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=defaultp
#SBATCH --constraint=delta|beta
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G

module load conda
conda activate crl
export PYTHONUNBUFFERED=1 WANDB_MODE=disabled OMP_NUM_THREADS=32
set -euo pipefail
: "${BATCH:?}" "${TAG:?}"

REPO=/nfs/scistore19/locatgrp/rcadei/artificial-causal-inference
ORIG=/nfs/scistore19/locatgrp/tmencatt/projects/neuralscience
OUT=$REPO/results/vision/ants/eci_v0_audit
BIG=/mnt/beegfs/locatgrp/shared/rcadei/eci_v0_audit/ants_v2/orig
cd "$REPO"; mkdir -p logs "$OUT" "$BIG"
STAGE=/localhome/$USER/$SLURM_JOB_ID
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/videos" "$STAGE/repo" "$STAGE/shim" "$STAGE/models"

# copy of the original code
rsync -a --exclude __pycache__ "$ORIG/src" "$STAGE/repo/"
sed -i "s#Path(\"/nfs/scistore19/locatgrp/tmencatt/projects/neuralscience/models\")#Path(\"$STAGE/models\")#" \
    "$STAGE/repo/src/neuralscience/runnable/s3-train_sae.py"
grep -q "$STAGE/models" "$STAGE/repo/src/neuralscience/runnable/s3-train_sae.py"
# s2 logs torch.cuda memory after every batch, which raises on a CPU-only node: drop that log call only
sed -i 's/^\(\s*\)print_gpu_mem(DEVICE)$/\1pass  # cuda-only memory log removed for CPU/' "$STAGE/repo/src/neuralscience/runnable/s2-encode_dataset.py"
grep -c "cuda-only memory log removed" "$STAGE/repo/src/neuralscience/runnable/s2-encode_dataset.py"
L=$REPO/logs/eci_v0_orig_${SLURM_JOB_ID}
printf 'class _F:\n    def __getitem__(self, k): return object\nFloat = _F()\n' > "$STAGE/shim/jaxtyping.py"
export PYTHONPATH="$STAGE/shim"

# data
python - <<'EOF' > "$STAGE/videos.txt"
import pandas as pd
e = pd.read_csv('data/ants/v2/experiment.csv'); print('\n'.join(e[e.valid == 1].observation_file))
EOF
while read -r f; do cp "data/ants/v2/observations/source/$f" "$STAGE/videos/"; done < "$STAGE/videos.txt"
M=$STAGE/manifest/rcadei/istant
python -u scripts/eci/v0_audit/orig_prepare.py --video-dir "$STAGE/videos" --manifest-dir "$M" --workers 32
mkdir -p "$M/SIGLIP" "$M/VIT" "$M/RESNET"   # s2's own skip rule: encode DINOV2 only

cd "$STAGE/repo"
echo "[$(date +%T)] s2"; python -u -m src.neuralscience.runnable.s2-encode_dataset --manifest_dir "$M" --root / --device cpu --batch_size 128 2>&1 | tr '\r' '\n' | grep -v "^Processed" > "${L}_s2.log"
python - "$M" <<'EOF'
import sys, numpy as np
a = np.load(f'{sys.argv[1]}/DINOV2/shards/shard_00000.npy'); b = np.load('/mnt/beegfs/locatgrp/shared/rcadei/eci_v0/ants_v2/tokens_batch_a.npy', mmap_mode='r')[:128]
print('their s2 tokens vs our v0 tokens (first 128 frames): shape', a.shape, 'max abs diff', float(np.abs(a - b).max()), 'mean abs', float(np.abs(b).mean()))
EOF
echo "[$(date +%T)] s3"; python -u -m src.neuralscience.runnable.s3-train_sae --manifest_dir "$M" --encoder_id DINOV2 --sae TOPK --k 20 --epochs 10 --batch_size "$BATCH" --device cpu 2>&1 | tr '\r' '\n' | grep -v "it/s\]$\|s/it\]$" > "${L}_s3.log"; grep "Epoch\[" "${L}_s3.log"
SAE_DIR=$(ls -d "$STAGE"/models/rcadei/istant/DINOV2/TOPK/*/ | head -1); SAE_DIR=${SAE_DIR%/}
echo "SAE dir $SAE_DIR"; cat "$SAE_DIR/metadata.json"
cp "$SAE_DIR/sae_weights.pth" "$BIG/sae_${TAG}.pth"; cp "$SAE_DIR/metadata.json" "$OUT/orig_${TAG}_sae_metadata.json"
for P in MEAN CLS; do
  echo "[$(date +%T)] s4 $P"; python -u -m src.neuralscience.runnable.s4-sae_encoding --sae_path "$SAE_DIR" --manifest_dir "$M" --pooler $P --device cpu --batch_size 128 2>&1 | tr '\r' '\n' | grep -v "^Processed\|it/s\]$\|s/it\]$" > "${L}_s4_$P.log"
  CODES="$M/SAE/TOPK/$P/DINOV2/$(basename "$SAE_DIR")/codes"
  python -u "$REPO/scripts/eci/v0_audit/orig_analyse.py" --orig-root "$STAGE/repo" --manifest-dir "$M" --codes-dir "$CODES" \
      --tag "${TAG}_${P}" --out "$OUT" --save-codes "$BIG/codes_${TAG}_${P}.npy"
done
echo "[$(date +%T)] DONE"
