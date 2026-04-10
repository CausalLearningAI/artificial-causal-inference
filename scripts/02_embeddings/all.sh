#!/bin/bash
#
# Submit embedding extraction jobs to gpu100 for all encoders × tokens.
# One SLURM job per (encoder, token) pair; each processes all experiments sequentially.
#
# Usage:
#   bash scripts/02_embeddings/all.sh                         # all encoders, both tokens
#   bash scripts/02_embeddings/all.sh siglip2                 # one encoder, both tokens
#   bash scripts/02_embeddings/all.sh siglip2 dinov2          # two encoders, both tokens
#
# To run a single encoder/token combo, use single.sh instead.
#
set -euo pipefail
cd "$(dirname "$0")/../.."

if [ $# -eq 0 ]; then
    ENCODERS=(siglip siglip2 dinov2 dinov3)
else
    ENCODERS=("$@")
fi

TOKENS=(class mean)

declare -A BATCH_SIZE=(
    [siglip]=80
    [siglip2]=112
    [dinov2]=192
    [dinov3]=192
)

mkdir -p logs /tmp/slurm_jobs_$$

for ENCODER in "${ENCODERS[@]}"; do
    BS=${BATCH_SIZE[$ENCODER]:-96}

    for TOKEN in "${TOKENS[@]}"; do
        JOB_NAME="emb_${ENCODER}_${TOKEN}"
        JOB_SCRIPT="/tmp/slurm_jobs_$$/job_${ENCODER}_${TOKEN}.sh"

        cat > "${JOB_SCRIPT}" << SLURM_SCRIPT
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=$(pwd)/logs/${JOB_NAME}_%j.out
#SBATCH --error=$(pwd)/logs/${JOB_NAME}_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:H100:1

set -euo pipefail
cd $(pwd)

module load conda
conda activate crl
export PYTHONUNBUFFERED=1

run_exp() {
    local SUBJECT=\$1 VERSION=\$2
    local OUT=dataset/\${SUBJECT}/\${VERSION}/embeddings/full/${ENCODER}/${TOKEN}
    if [ -d "\${OUT}" ] && [ "\$(ls -A "\${OUT}" 2>/dev/null)" ]; then
        echo "[SKIP] \${SUBJECT}/\${VERSION} — already exists at \${OUT}"
        return 0
    fi
    echo "[START] \${SUBJECT}/\${VERSION} — encoder=${ENCODER} token=${TOKEN}"
    local T0=\$(date +%s)
    python -u src/embedding/get_embeddings.py \
        experiment="\${SUBJECT}/\${VERSION}" \
        encoder="${ENCODER}" \
        token="${TOKEN}" \
        batch_size="${BS}" \
        num_workers=8 \
        device=cuda
    echo "[DONE] \${SUBJECT}/\${VERSION} in \$(( \$(date +%s) - T0 ))s"
}

run_exp ants v1
run_exp ants v2
run_exp ants v3
run_exp ants v4
run_exp ants v5
run_exp mice v1
run_exp mice v2
SLURM_SCRIPT

        JOB_ID=$(sbatch "${JOB_SCRIPT}" | awk '{print $NF}')
        echo "Submitted ${JOB_NAME} → job ${JOB_ID}  (gpu100, batch_size=${BS})"
    done
done

rm -rf /tmp/slurm_jobs_$$
