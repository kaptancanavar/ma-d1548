#!/bin/bash
#SBATCH --job-name=VXM_TRY10_LR6e-5_BASE
#SBATCH --partition=student
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=2-00:00:00
#SBATCH --mem=24G
#SBATCH --output=/home/students/studunals1/biopsy_project/vxm/LOSS_axial_new_sweep/logs/try10_lr6e-5_base.%J.out
#SBATCH --error=/home/students/studunals1/biopsy_project/vxm/LOSS_axial_new_sweep/logs/try10_lr6e-5_base.%J.err

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

SCRIPT="/home/students/studunals1/biopsy_project/vxm/LOSS_axial_new_sweep/train_vxm_lesion_axis_mse_mmheatmap_sweep.py"

"$PY" "$SCRIPT" \
  --train-csv /mnt/ceph/vol_02_home_students/studunals1/biopsy_project/vxm/Analysis/train_lesion_0.csv \
  --val-csv   /mnt/ceph/vol_02_home_students/studunals1/biopsy_project/vxm/Analysis/val_lesion_0.csv \
  --nii-json-mapping-csv /home/students/studunals1/biopsy_project/vxm/Analysis/nii_json_mapping.csv \
  --out-root  /home/students/studunals1/biopsy_project/vxm/LOSS_axial_new_sweep/run \
  --epochs 80 \
  --batch-size 1 \
  --num-workers 4 \
  --image-loss ncc \
  --lr 6e-5 \
  --sim-weight 0.28 \
  --reg-weight 0.12 \
  --grad-downsample 1 \
  --int-steps 0 \
  --input-shape 224 224 96 \
  --lesion-loss-mode axis_mse \
  --lesion-w-mm 2.0 1.0 1.0 \
  --lesion-weight 0.8 \
  --lesion-sigma-mm 2.0 \
  --lesion-beta 2.0 \
  --skip-missing-lesion \
  --early-stop-metric lesion_err \
  --early-stop-patience 15 \
  --early-stop-min-delta 0.0005 \
  --log-z-slice-error \
  --save-every 10 \
  --wandb-project axial_loss_sweep \
  --exp-id TRY10_lr6e-5_baselineLike

STATUS=$?
echo "Ende:  $(date)"
echo "Exit:  $STATUS"
exit $STATUS
