#!/bin/bash
#
# Deployed-model annotations and checks of a new ants version (new_version_report.py):
# annotate -> rates -> diagnostic. MLP inference only, so it runs on CPU (no GPU queue).
#
# Usage: VERSION=v4_infested REFS="v4 v5" sbatch [--dependency=afterok:<emb jobs>] scripts/05_deploy/job_new_version_deploy.sh
#
#SBATCH --job-name=deploy_new
#SBATCH --output=logs/deploy_new_%j.out
#SBATCH --error=logs/deploy_new_%j.err
#SBATCH --time=03:00:00
#SBATCH --partition=defaultp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(git rev-parse --show-toplevel)}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
PY=${CONDA_ENVS:-$HOME/.conda/envs}/crl/bin/python
V=${VERSION:?}
for cmd in annotate rates "diagnostic --refs ${REFS:-v4 v5}"; do
    t=$(date +%s)
    $PY -u scripts/05_deploy/new_version_report.py --version $V $cmd
    echo "[$cmd done in $(( $(date +%s) - t ))s]"
done
