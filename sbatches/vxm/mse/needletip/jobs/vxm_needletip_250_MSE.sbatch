#!/bin/bash
#SBATCH --job-name=VXM_MSE_NEEDLETIP
#SBATCH --partition=week
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=2-00:00:00
#SBATCH --mem=24G
#SBATCH --output=/home/students/studunals1/biopsy_project/vxm/mse/needletip/logs/train_vxm_MSE_needletip.%J.out
#SBATCH --error=/home/students/studunals1/biopsy_project/vxm/mse/needletip/logs/train_vxm_MSE_needletip.%J.err

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

# Arbeitsverzeichnis für Logs/Run
cd /home/students/studunals1/biopsy_project/vxm/mse/needletip
mkdir -p logs run

# --- RUN (MSE sweep run) ---
"$PY" /home/students/studunals1/biopsy_project/vxm/scripts/vxm_train_lesion.py \
  --train-csv /mnt/ceph/vol_02_home_students/studunals1/biopsy_project/vxm/Analysis/train_needletip_0.csv \
  --val-csv   /mnt/ceph/vol_02_home_students/studunals1/biopsy_project/vxm/Analysis/val_needletip_0.csv \
  --out-root  /home/students/studunals1/biopsy_project/vxm/mse/needletip/run \
  --epochs 250 \
  --batch-size 1 \
  --num-workers 4 \
  --image-loss mse \
  --lr 0.0002859914039503391 \
  --sim-weight 1.0 \
  --reg-weight 0.003020618766354234 \
  --grad-downsample 1 \
  --int-steps 0 \
  --input-shape 224 224 96 \
  --exp-id MSE_needletip_250

STATUS=$?
echo "Ende:  $(date)"
echo "Exit:  $STATUS"
exit $STATUS
