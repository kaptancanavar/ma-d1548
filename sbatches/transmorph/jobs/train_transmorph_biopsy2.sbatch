#!/bin/bash
#SBATCH --job-name=TM_NCC_LESION
#SBATCH --partition=week
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=2-00:00:00
#SBATCH --mem=32G
#SBATCH --output=/home/students/studunals1/biopsy_project/transmorph/logs/train_tm_ncc_lesion.%J.out
#SBATCH --error=/home/students/studunals1/biopsy_project/transmorph/logs/train_tm_ncc_lesion.%J.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "Host:  $(hostname)"
echo "Job:   ${SLURM_JOB_ID:-manual}"
echo "Start: $(date)"
echo "---- ENV ----"

PY="/home/students/studunals1/.conda/envs/biopsy_env/bin/python"
"$PY" -V
echo "Using: $PY"
echo "------------"

PROJECT_ROOT="/home/students/studunals1/biopsy_project/transmorph"
SCRIPT="${PROJECT_ROOT}/scripts/train_transmorph_biopsy.py"
TRANSMORPH_REPO="${PROJECT_ROOT}/repo/TransMorph_Transformer_for_Medical_Image_Registration"

mkdir -p "${PROJECT_ROOT}/logs"
mkdir -p "${PROJECT_ROOT}/run"

cd "${PROJECT_ROOT}"

"$PY" "$SCRIPT" \
  --transmorph-repo "$TRANSMORPH_REPO" \
  --train-csv /mnt/ceph/vol_02_home_students/studunals1/biopsy_project/vxm/Analysis/train_lesion_0.csv \
  --val-csv   /mnt/ceph/vol_02_home_students/studunals1/biopsy_project/vxm/Analysis/val_lesion_0.csv \
  --out-root  "${PROJECT_ROOT}/run" \
  --epochs 50 \
  --batch-size 1 \
  --num-workers 4 \
  --image-loss ncc \
  --lr 1e-4 \
  --sim-weight 1.0 \
  --reg-weight 0.1 \
  --input-shape 224 224 96 \
  --window-size 7 7 3 \
  --transmorph-variant small \
  --save-every 10 \
  --exp-id NCC_lesion_250

STATUS=$?
echo "Ende:  $(date)"
echo "Exit:  $STATUS"
exit $STATUS