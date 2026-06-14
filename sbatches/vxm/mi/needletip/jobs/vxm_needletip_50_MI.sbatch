#!/bin/bash
#SBATCH --job-name=VXM_MI_NEEDLETIP
#SBATCH --partition=week
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=0-02:00:00
#SBATCH --mem=24G
#SBATCH --output=/home/students/studunals1/biopsy_project/vxm/mi/needletip/logs/train_vxm_MI_needletip.%J.out
#SBATCH --error=/home/students/studunals1/biopsy_project/vxm/mi/needletip/logs/train_vxm_MI_needletip.%J.err

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
cd /home/students/studunals1/biopsy_project/vxm/mi/needletip
mkdir -p logs run

# --- RUN (MI sweep run) ---
"$PY" /home/students/studunals1/biopsy_project/vxm/scripts/vxm_train_lesion.py \
  --train-csv /mnt/ceph/vol_02_home_students/studunals1/biopsy_project/vxm/Analysis/train_needletip_0.csv \
  --val-csv   /mnt/ceph/vol_02_home_students/studunals1/biopsy_project/vxm/Analysis/val_needletip_0.csv \
  --out-root  /home/students/studunals1/biopsy_project/vxm/mi/needletip/run \
  --epochs 50 \
  --batch-size 1 \
  --num-workers 4 \
  --image-loss mi \
  --lr 0.000271988405420567 \
  --sim-weight 1.0 \
  --reg-weight 0.3758583388762706 \
  --grad-downsample 1 \
  --int-steps 0 \
  --mi-bins 32 \
  --mi-sigma 0.2 \
  --mi-samples 5000 \
  --mi-clip 5.0 \
  --input-shape 224 224 96 \
  --exp-id MI_needletip_50

STATUS=$?
echo "Ende:  $(date)"
echo "Exit:  $STATUS"
exit $STATUS
