#!/bin/bash
#SBATCH --job-name=GMA_Inf_Lesion
#SBATCH --partition=week
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=0-06:00:00
#SBATCH --mem=16G
#SBATCH --output=/home/students/studunals1/biopsy_project/gmaraft3d/logs/%x.%j.out
#SBATCH --error=/home/students/studunals1/biopsy_project/gmaraft3d/logs/%x.%j.err

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
echo "Using PY: $PY"
"$PY" -V

ROOT="/home/students/studunals1/biopsy_project/gmaraft3d"
cd "$ROOT/4Louisa"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$ROOT/logs"
OUT_ROOT="$ROOT/inference_lesion/gmaraft_bestmodel"
mkdir -p "$OUT_ROOT"

CSV="/home/students/studunals1/biopsy_project/vxm/Analysis/val_lesion_0.csv"
CKPT="/mnt/ceph/vol_02_home_students/studunals1/biopsy_project/gmaraft3d/experiments_gmaraft/260303_014846_silver-capybara-67_y2pbat78/checkpoints/best_model.pth"

"$PY" infer_breast_gmaraft3d.py \
  --csv "$CSV" \
  --checkpoint "$CKPT" \
  --out-root "$OUT_ROOT" \
  --input-shape 224 224 96 \
  --train-shape 96 96 64 \
  --batch-size 1 \
  --num-workers 2 \
  --save-flow

STATUS=$?
echo "Ende:  $(date)"
echo "Exit:  $STATUS"
exit $STATUS
