#!/bin/bash
#SBATCH --job-name=VXM_TRY2_W2_1_1
#SBATCH --partition=student
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=0-06:00:00
#SBATCH --mem=24G
#SBATCH --output=/home/students/studunals1/biopsy_project/vxm/LOSS_axial_new_sweep/logs/try2_w2_1_1.%J.out
#SBATCH --error=/home/students/studunals1/biopsy_project/vxm/LOSS_axial_new_sweep/logs/try2_w2_1_1.%J.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "Host:  $(hostname)"
echo "Job:   ${SLURM_JOB_ID:-manual}"
echo "Start: $(date)"

PY="/home/students/studunals1/.conda/envs/biopsy_env/bin/python"
"$PY" -V
echo "Using: $PY"

cd /home/students/studunals1/biopsy_project/vxm/LOSS_axial_new_sweep
mkdir -p logs run

"$PY" /home/students/studunals1/biopsy_project/vxm/LOSS_axial_new_sweep/train_vxm_lesion_axis_mse_mmheatmap_sweep.py \
  --train-csv /mnt/ceph/vol_02_home_students/studunals1/biopsy_project/vxm/Analysis/train_lesion_0.csv \
  --val-csv   /mnt/ceph/vol_02_home_students/studunals1/biopsy_project/vxm/Analysis/val_lesion_0.csv \
  --nii-json-mapping-csv /home/students/studunals1/biopsy_project/vxm/Analysis/nii_json_mapping.csv \
  --out-root  /home/students/studunals1/biopsy_project/vxm/LOSS_axial_new_sweep/run \
  --epochs 50 \
  --batch-size 1 \
  --num-workers 4 \
  --image-loss ncc \
  --lr 3.0e-05 \
  --sim-weight 0.25 \
  --reg-weight 0.15 \
  --grad-downsample 1 \
  --int-steps 0 \
  --input-shape 224 224 96 \
  --lesion-loss-mode axis_mse \
  --lesion-w-mm 2.0 1.0 1.0 \
  --lesion-weight 1.05 \
  --lesion-sigma-mm 2.0 \
  --lesion-beta 4 \
  --skip-missing-lesion \
  --log-z-slice-error \
  --wandb-project axial_loss_sweep \
  --early-stop-metric lesion_err \
  --early-stop-patience 12 \
  --early-stop-min-delta 0.001 \
  --save-every 10 \
  --exp-id TRY2_W2_1_1

STATUS=$?
echo "Ende:  $(date)"
echo "Exit:  $STATUS"
exit $STATUS
