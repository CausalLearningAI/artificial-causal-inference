#!/bin/bash
#
# Stages of the 30-min extension of an ants version (option B: minutes 0-10 kept from the
# 10-min build). Each call submits one Slurm job; run them in this order, checking each log:
#
#   python scripts/05_deploy/extend_30min.py --version v5 archive          # login node: renames only
#   python scripts/05_deploy/add_annotation_window.py v5 --apply           # end_frame + annotation_end_frame
#   python scripts/05_deploy/extend_30min.py --version v5 obs-list
#   VERSION=v5 sbatch --array=0-189%30 scripts/05_deploy/job_extend_30min.sh   # S0+S1a+S3a+S3b per video
#   VERSION=v5 bash scripts/05_deploy/extend_30min_stages.sh s1            # metadata, annotations, HF
#   VERSION=v5 bash scripts/05_deploy/extend_30min_stages.sh s3a           # prefix verify, tracking splice + seam
#   VERSION=v5 bash scripts/05_deploy/extend_30min_stages.sh s3c blue      # POV embeddings (and yellow)
#   VERSION=v5 bash scripts/05_deploy/extend_30min_stages.sh s3d blue      # embedding splice (and yellow)
#   then the acceptance gate (fingerprint_datasets.py, deploy_model.py --eval-only), and
#   VERSION=v5 bash scripts/05_deploy/extend_30min_stages.sh s5            # generate_annotations
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
        sbatch --parsable --job-name="$name" --output="logs/${name}_%j.out" --error="logs/${name}_%j.err" "$@"
}

case "$STAGE" in
s1) submit "ext_s1_${VERSION}" --partition=defaultp --cpus-per-task=8 --mem=64G --time=10:00:00 <<EOF
\$PY -u src/data/get_metadata.py experiment=ants/$VERSION
\$PY -u -m src.dataset.get_annotations experiment=ants/$VERSION
\$PY -u scripts/05_deploy/extend_30min.py --version $VERSION verify-annotations
\$PY -u src/dataset/get_dataset.py experiment=ants/$VERSION
EOF
;;
s3a) submit "ext_s3a_${VERSION}" --partition=defaultp --cpus-per-task=4 --mem=16G --time=03:00:00 <<EOF
\$PY -u scripts/05_deploy/extend_30min.py --version $VERSION verify-prefix
\$PY -u scripts/05_deploy/extend_30min.py --version $VERSION splice-tracking
EOF
;;
s3c) submit "ext_s3c_${VERSION}_${IDENT}" --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=48G --time=24:00:00 <<EOF
\$PY -u src/embedding/get_embeddings.py experiment=ants/$VERSION encoder=dinov2 token=class \
    batch_size=192 num_workers=8 device=cuda +frame_type=pov +pov_identity=$IDENT overwrite.embeddings=false
EOF
;;
s3d) submit "ext_s3d_${VERSION}_${IDENT}" --partition=defaultp --cpus-per-task=4 --mem=64G --time=04:00:00 <<EOF
ANN=\$(\$PY -c "import pandas as pd; e=pd.read_csv('data/ants/$VERSION/experiment.csv'); print(' '.join(e[(e.valid==1)&e.annotation_file.notna()].observation_id))")
\$PY -u scripts/05_deploy/extend_30min.py --version $VERSION splice-embeddings --identity $IDENT --control-obs \$ANN
EOF
;;
s5) submit "ext_s5_${VERSION}" --partition=gpu --gres=gpu:1 --cpus-per-task=4 --mem=64G --time=02:00:00 <<EOF
\$PY -u src/ppci/generate_annotations.py --version $VERSION
EOF
;;
*) echo "unknown stage $STAGE (s1 s3a s3c s3d s5)"; exit 1 ;;
esac
