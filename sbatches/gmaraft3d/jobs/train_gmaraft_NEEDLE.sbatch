#!/bin/bash
#SBATCH --job-name=GMA_Needle
#SBATCH --partition=week
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=2-00:00:00
#SBATCH --mem=32G
#SBATCH --output=/home/students/studunals1/biopsy_project/gmaraft3d/logs/Needle%x.%j.out
#SBATCH --error=/home/students/studunals1/biopsy_project/gmaraft3d/logs/Needle%x.%j.err

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
WORKDIR="$ROOT/4Louisa"
cd "$WORKDIR"

# ensure local imports work
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$ROOT/logs"
mkdir -p "$ROOT/experiments_gmaraft"

# W&B (project/group; run name empty -> wandb auto name)
export WANDB_PROJECT="breast-biopsy-gmaraft3d"
export WANDB_ENTITY="st148315-university-of-stuttgart"
export WANDB_MODE="online"

"$PY" train_breast_gmaraft3d_LOSS.py \
  --train-csv /mnt/ceph/vol_02_home_students/studunals1/biopsy_project/vxm/Analysis/train_needletip_0.csv \
  --val-csv   /mnt/ceph/vol_02_home_students/studunals1/biopsy_project/vxm/Analysis/val_needletip_0.csv \
  --out-root  /home/students/studunals1/biopsy_project/gmaraft3d/experiments_gmaraft \
  --epochs 50 \
  --batch-size 1 \
  --num-workers 4 \
  --lr 0.0002 \
  --weight-decay 5e-05 \
  --input-shape 224 224 96 \
  --train-shape 96 96 64 \
  --sim-weight 1.0 \
  --reg-weight 0.01 \
  --gamma 0.85 \
  --zclamp 5 \
  --flow-clamp 20 \
  --grad-clip 1 \
  --use-amp \
  --save-every 10 \
  --seed 42 \
  --wandb-project breast-biopsy-gmaraft3d \
  --wandb-group GMA_NCC_SWEEP_2 \
  --wandb-mode online

STATUS=$?
echo "Ende:  $(date)"
echo "Exit:  $STATUS"
exit $STATUS
