#!/bin/bash
#SBATCH --job-name=VXM_MI_TEST
#SBATCH --partition=week
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=0-02:00:00
#SBATCH --mem=24G
#SBATCH --output=/home/students/studunals1/biopsy_project/vxm/mi/lesion/logs/train_vxm_MI_TEST.%J.out
#SBATCH --error=/home/students/studunals1/biopsy_project/vxm/mi/lesion/logs/train_vxm_MI_TEST.%J.err

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
cd /home/students/studunals1/biopsy_project/vxm/mi/lesion
mkdir -p logs run

# --- RUN (MI smoke test) ---
"$PY" /home/students/studunals1/biopsy_project/vxm/scripts/vxm_train_lesion.py \
  --train-csv /home/students/studunals1/biopsy_project/vxm/Analysis/train_lesion_0.csv \
  --val-csv   /home/students/studunals1/biopsy_project/vxm/Analysis/val_lesion_0.csv \
  --out-root  /home/students/studunals1/biopsy_project/vxm/mi/lesion/run \
  --epochs 3 \
  --batch-size 1 \
  --num-workers 4 \
  --image-loss mi \
  --lr 1e-4 \
  --reg-weight 1.0 \
  --mi-bins 16 \
  --mi-sigma 0.2 \
  --mi-samples 5000 \
  --mi-clip 5.0 \
  --input-shape 224 224 96 \
  --exp-id MI_TEST

STATUS=$?
echo "Ende:  $(date)"
echo "Exit:  $STATUS"
exit $STATUS
