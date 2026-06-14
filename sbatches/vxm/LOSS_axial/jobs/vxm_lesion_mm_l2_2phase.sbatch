#!/bin/bash
#SBATCH --job-name=VXM_LES_MM_L2_2PH_AW_1_1_0p5
#SBATCH --partition=week
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00
#SBATCH --mem=24G
#SBATCH --output=/home/students/studunals1/biopsy_project/vxm/LOSS_axial/logs/vxm_les_mm_l2_2phase_axisw_1_1_0p5.%J.out
#SBATCH --error=/home/students/studunals1/biopsy_project/vxm/LOSS_axial/logs/vxm_les_mm_l2_2phase_axisw_1_1_0p5.%J.err

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

cd /home/students/studunals1/biopsy_project/vxm/LOSS_axial
mkdir -p logs run

"$PY" /home/students/studunals1/biopsy_project/vxm/LOSS_axial/vxm_lesion_mm_l2_2phase_axisw.py \
  --train-csv /mnt/ceph/vol_02_home_students/studunals1/biopsy_project/vxm/Analysis/train_lesion_0.csv \
  --val-csv   /mnt/ceph/vol_02_home_students/studunals1/biopsy_project/vxm/Analysis/val_lesion_0.csv \
  --nii-json-mapping-csv /home/students/studunals1/biopsy_project/vxm/Analysis/nii_json_mapping.csv \
  --out-root  /home/students/studunals1/biopsy_project/vxm/LOSS_axial/run \
  --epochs 120 \
  --batch-size 1 \
  --num-workers 4 \
  --image-loss ncc \
  --lr 0.00005 \
  --sim-weight 0.3 \
  --reg-weight 0.08 \
  --grad-downsample 1 \
  --int-steps 0 \
  --input-shape 224 224 96 \
  --lesion-loss-mode l2 \
  --lesion-w-mm 1 1 0.5 \
  --lesion-weight 0.8 \
  --lesion-sigma-vox 2.0 \
  --lesion-beta 2.0 \
  --phase2-epoch 50 \
  --sim-weight2 0.15 \
  --reg-weight2 0.05 \
  --lesion-weight2 2.0 \
  --lesion-sigma-vox2 1.5 \
  --lesion-beta2 4.0 \
  --skip-missing-lesion \
  --early-stop-metric lesion_err \
  --early-stop-patience 30 \
  --early-stop-min-delta 0.001 \
  --save-every 10 \
  --exp-id LES_MM_L2_2PH_axisw1_1_0p5_ncc_p2e50_sw0p3to0p15_rw0p08to0p05_lw0p8to2p0_sig2to1p5_b2to4_lr5e-5_e120

STATUS=$?
echo "Ende:  $(date)"
echo "Exit:  $STATUS"
exit $STATUS
