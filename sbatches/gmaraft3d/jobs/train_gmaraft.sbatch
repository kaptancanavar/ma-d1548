#!/bin/bash
#SBATCH --job-name=GMA_E0_NCC_REG001_96x96x64
#SBATCH --partition=week
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=2-00:00:00
#SBATCH --mem=32G
#SBATCH --output=/home/students/studunals1/biopsy_project/gmaraft3d/logs/train_gma.%J.out
#SBATCH --error=/home/students/studunals1/biopsy_project/gmaraft3d/logs/train_gma.%J.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

echo "Host:  $(hostname)"
echo "Job:   ${SLURM_JOB_ID:-manual}"
echo "Start: $(date)"

PY="/home/students/studunals1/.conda/envs/biopsy_env/bin/python"
echo "Using: $PY"
"$PY" -V

ROOT="/home/students/studunals1/biopsy_project/gmaraft3d"
cd "$ROOT/4Louisa"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

mkdir -p "$ROOT/logs" "$ROOT/experiments_gmaraft"

"$PY" train_breast_gmaraft3d.py \
  --train-csv /mnt/ceph/vol_02_home_students/studunals1/biopsy_project/vxm/Analysis/train_lesion_0.csv \
  --val-csv   /mnt/ceph/vol_02_home_students/studunals1/biopsy_project/vxm/Analysis/val_lesion_0.csv \
  --out-root  "$ROOT/experiments_gmaraft" \
  --epochs 50 \
  --batch-size 1 \
  --num-workers 4 \
  --lr 2e-4 \
  --input-shape 224 224 96 \
  --train-shape 96 96 64 \
  --sim-weight 1.0 \
  --reg-weight 0.01 \
  --save-every 10

STATUS=$?
echo "Ende:  $(date)"
echo "Exit:  $STATUS"
exit $STATUS
