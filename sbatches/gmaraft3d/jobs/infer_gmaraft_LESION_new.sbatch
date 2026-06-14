#!/bin/bash
#SBATCH --job-name=GMA_Inf_Lesion_New
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

OUT_ROOT="$ROOT/inference_lesion/gmaraft_bestmodel_new"
mkdir -p "$OUT_ROOT"

CSV="/home/students/studunals1/biopsy_project/vxm/Analysis/test_lesion_0.csv"
CKPT="$ROOT/experiments_gmaraft/260305_220909_peach-sweep-24_fsnt5vgw/checkpoints/best_model.pth"
SCRIPT="infer_breast_gmaraft3d_axial.py"

"$PY" "$SCRIPT" \
  --csv "$CSV" \
  --checkpoint "$CKPT" \
  --out-root "$OUT_ROOT" \
  --input-shape 224 224 96 \
  --train-shape 96 96 64 \
  --zclamp 5.0 \
  --flow-clamp 20.0 \
  --save-flow

STATUS=$?
echo "Ende:  $(date)"
echo "Exit:  $STATUS"
exit $STATUS