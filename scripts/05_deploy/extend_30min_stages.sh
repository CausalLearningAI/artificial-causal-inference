#!/bin/bash
#
# Stages of the 30-min extension of an ants version (option B: minutes 0-10 kept from the
# 10-min build). Each call submits one Slurm job; run them in this order, checking each log:
#
#   python scripts/05_deploy/extend_30min.py --version v5 archive          # login node: renames only
#     (v3/v4: the 30-min sources then go into data/ants/<v>/observations/source by mv)
#   python scripts/05_deploy/add_annotation_window.py v5 --apply           # end_frame + annotation_end_frame
#   python scripts/05_deploy/extend_30min.py --version v5 obs-list [--n-check 20]
#   VERSION=v5 sbatch --array=0-189%30 scripts/05_deploy/job_extend_30min.sh   # S0+S1a+S3a+S3b per video
#   VERSION=v5 bash scripts/05_deploy/extend_30min_stages.sh s1            # metadata, annotations
#   VERSION=v5 bash scripts/05_deploy/extend_30min_stages.sh s1hf          # HF dataset (parallel to s3*)
#   VERSION=v5 bash scripts/05_deploy/extend_30min_stages.sh s3a           # prefix verify, tracking splice + seam
#   VERSION=v5 bash scripts/05_deploy/extend_30min_stages.sh s3c blue      # POV embeddings (and yellow)
#   VERSION=v5 bash scripts/05_deploy/extend_30min_stages.sh s3d blue      # embedding splice (and yellow)
#   VERSION=v5 bash scripts/05_deploy/extend_30min_stages.sh check5        # drift of re-processed minutes 0-10
#   then the acceptance gate (fingerprint_datasets.py, deploy_model.py --eval-only), and
#   VERSION=v5 bash scripts/05_deploy/extend_30min_stages.sh s5            # generate_annotations
#   VERSION=v5 bash scripts/05_deploy/extend_30min_stages.sh quality       # check_30min_quality
# Extra sbatch options (e.g. --dependency=afterok:<id>) go in SBATCH_EXTRA.
#
# ffmpeg comes from FFMPEG_ENV (6.1.1 + openh264 2.1.1, the build that encoded the 10-min clips),
# python from the crl env. Scratch (HF cache) lives on node-local disk and is removed on exit.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

VERSION=${VERSION:-v5}
ENVS=${CONDA_ENVS:-$HOME/.conda/envs}
FFMPEG_ENV=${FFMPEG_ENV:-$ENVS/ffmpeg611}
STAGE=$1; IDENT=${2:-}
mkdir -p logs

PRE="set -euo pipefail
cd $PWD
export PATH=$FFMPEG_ENV/bin:$ENVS/crl/bin:\$PATH PYTHONUNBUFFERED=1
S=/localhome/\$USER/\$SLURM_JOB_ID; mkdir -p \$S 2>/dev/null || S=\${TMPDIR:-/tmp}/\$USER/\$SLURM_JOB_ID; mkdir -p \$S
export HF_DATASETS_CACHE=\$S/hf_cache; trap 'rm -rf \$S' EXIT
PY=$ENVS/crl/bin/python
echo \"node=\$(hostname) \$(ffmpeg -version | head -1)\""

submit() {  # name, sbatch options..., then the body on stdin
    local name=$1; shift
    printf '#!/bin/bash\n%s\n%s\n' "$PRE" "$(cat)" |
        sbatch --parsable --job-name="$name" --output="logs/${name}_%j.out" --error="logs/${name}_%j.err" \
            ${SBATCH_EXTRA:-} "$@"
}
GPULOG='nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -l 60 > logs/gpu_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.csv & GPUMON=$!'

case "$STAGE" in
s1) submit "ext_s1_${VERSION}" --partition=defaultp --cpus-per-task=8 --mem=64G --time=10:00:00 <<EOF
\$PY -u src/data/get_metadata.py experiment=ants/$VERSION
\$PY -u -m src.dataset.get_annotations experiment=ants/$VERSION
\$PY -u scripts/05_deploy/extend_30min.py --version $VERSION verify-annotations
EOF
;;
s1hf) submit "ext_s1hf_${VERSION}" --partition=defaultp --cpus-per-task=32 --mem=64G --time=10:00:00 <<EOF
\$PY -u scripts/05_deploy/extend_30min.py --version $VERSION build-hf --num-proc 32 2> >(grep -v "%|" >&2)
EOF
;;
s3a) submit "ext_s3a_${VERSION}" --partition=defaultp --cpus-per-task=4 --mem=16G --time=03:00:00 <<EOF
\$PY -u scripts/05_deploy/extend_30min.py --version $VERSION verify-prefix
\$PY -u scripts/05_deploy/extend_30min.py --version $VERSION splice-tracking
EOF
;;
s3c) submit "ext_s3c_${VERSION}_${IDENT}" --partition=gpu --gres=gpu:1 --cpus-per-task=16 --mem=64G --time=24:00:00 <<EOF
$GPULOG
\$PY -u src/embedding/get_embeddings.py experiment=ants/$VERSION encoder=dinov2 token=class \
    batch_size=192 num_workers=16 device=cuda +frame_type=pov +pov_identity=$IDENT overwrite.embeddings=false \
    2> >(grep -v "frame/s\]" >&2)
kill \$GPUMON
EOF
;;
s3d) submit "ext_s3d_${VERSION}_${IDENT}" --partition=defaultp --cpus-per-task=4 --mem=64G --time=04:00:00 <<EOF
\$PY -u scripts/05_deploy/extend_30min.py --version $VERSION splice-embeddings --identity $IDENT \
    --control-obs \$(cat results/ppci/ants/regress_30min/${VERSION}_check_obs.txt) 2> >(grep -v "%|" >&2)
EOF
;;
check5) submit "ext_check5_${VERSION}" --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=96G --time=02:00:00 <<EOF
$GPULOG
STAGE_DIR=\$S/check5 \$PY -u scripts/05_deploy/check5_reprocess.py --version $VERSION 2> >(grep -v "frame/s\]" >&2)
kill \$GPUMON
EOF
;;
s5) submit "ext_s5_${VERSION}" --partition=gpu --gres=gpu:1 --cpus-per-task=4 --mem=64G --time=02:00:00 <<EOF
$GPULOG
\$PY -u src/ppci/generate_annotations.py --version $VERSION
kill \$GPUMON
EOF
;;
quality) submit "ext_quality_${VERSION}" --partition=defaultp --cpus-per-task=4 --mem=32G --time=02:00:00 <<EOF
\$PY -u scripts/05_deploy/check_30min_quality.py --version $VERSION \
    --sheet-obs \$(head -n 5 results/ppci/ants/regress_30min/${VERSION}_check_obs.txt)
EOF
;;
*) echo "unknown stage $STAGE (s1 s1hf s3a s3c s3d check5 s5 quality)"; exit 1 ;;
esac
