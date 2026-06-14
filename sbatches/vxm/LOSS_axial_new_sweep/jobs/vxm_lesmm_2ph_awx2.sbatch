#!/bin/bash
#SBATCH --job-name=VXM_LES_MM_2PH_AWX2_SIG2to1p5_B2to4
#SBATCH --partition=student
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=2-00:00:00
#SBATCH --mem=24G
#SBATCH --output=/home/students/studunals1/biopsy_project/vxm/LOSS_axial_new_sweep/logs/vxm_les_mm_2ph_awx2.%J.out
#SBATCH --error=/home/students/studunals1/biopsy_project/vxm/LOSS_axial_new_sweep/logs/vxm_les_mm_2ph_awx2.%J.err

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

# >>> Pfad zu deinem Skript ggf. anpassen:
SCRIPT="/home/students/studunals1/biopsy_project/vxm/LOSS_axial_new_sweep/train_vxm_lesion_axis_mse_mmheatmap_sweep.py"

"$PY" "$SCRIPT" \
  --train-csv /mnt/ceph/vol_02_home_students/studunals1/biopsy_project/vxm/Analysis/train_lesion_0.csv \
  --val-csv   /mnt/ceph/vol_02_home_students/studunals1/biopsy_project/vxm/Analysis/val_lesion_0.csv \
  --nii-json-mapping-csv /home/students/studunals1/biopsy_project/vxm/Analysis/nii_json_mapping.csv \
  --out-root  /home/students/studunals1/biopsy_project/vxm/LOSS_axial_new_sweep/run \
  --epochs 50 \
  --batch-size 1 \
  --num-workers 4 \
  --image-loss ncc \
  --lr 3e-5 \
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
  --phase2-epoch 21 \
  --sim-weight2 0.18 \
  --reg-weight2 0.08 \
  --lesion-weight2 2.0 \
  --lesion-sigma-mm2 1.5 \
  --lesion-beta2 4.0 \
  --skip-missing-lesion \
  --early-stop-metric lesion_err \
  --early-stop-patience 30 \
  --early-stop-min-delta 0.0005 \
  --log-z-slice-error \
  --save-every 10 \
  --wandb-project axial_loss_sweep \
  --exp-id LES_MM_2PH_awx2_sig2to1p5_b2to4_ncc_lr3e-5

STATUS=$?
echo "Ende:  $(date)"
echo "Exit:  $STATUS"
exit $STATUS
