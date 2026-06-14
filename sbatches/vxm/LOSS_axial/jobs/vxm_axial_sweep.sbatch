#!/bin/bash
#SBATCH --job-name=VXM_AXIAL_SWEEP
#SBATCH --partition=week
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00
#SBATCH --mem=24G
#SBATCH --array=0-5
#SBATCH --output=/home/students/studunals1/biopsy_project/vxm/LOSS_axial/logs/vxm_axial_sweep.%A_%a.out
#SBATCH --error=/home/students/studunals1/biopsy_project/vxm/LOSS_axial/logs/vxm_axial_sweep.%A_%a.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "Host:  $(hostname)"
echo "Job:   ${SLURM_JOB_ID:-manual}"
echo "Array: ${SLURM_ARRAY_JOB_ID:-na}_${SLURM_ARRAY_TASK_ID:-na}"
echo "Start: $(date)"

PY="/home/students/studunals1/.conda/envs/biopsy_env/bin/python"
"$PY" -V
echo "Using: $PY"

# Projekt-Ordner wie bei dir
cd /home/students/studunals1/biopsy_project/vxm/LOSS_axial
mkdir -p logs run

SCRIPT="/home/students/studunals1/biopsy_project/vxm/LOSS_axial/vxm_lesion_mm_l2_2phase_axisw.py"

TRAIN_CSV="/mnt/ceph/vol_02_home_students/studunals1/biopsy_project/vxm/Analysis/train_lesion_0.csv"
VAL_CSV="/mnt/ceph/vol_02_home_students/studunals1/biopsy_project/vxm/Analysis/val_lesion_0.csv"
MAP_CSV="/home/students/studunals1/biopsy_project/vxm/Analysis/nii_json_mapping.csv"
OUT_ROOT="/home/students/studunals1/biopsy_project/vxm/LOSS_axial/run"

# ---- Basis-Argumente: nah an deinem funktionierenden Beispiel ----
BASE_ARGS=(
  --train-csv "$TRAIN_CSV"
  --val-csv   "$VAL_CSV"
  --nii-json-mapping-csv "$MAP_CSV"
  --out-root  "$OUT_ROOT"
  --epochs 120
  --batch-size 1
  --num-workers 4
  --image-loss ncc
  --lr 0.00005
  --sim-weight 0.3
  --reg-weight 0.08
  --grad-downsample 1
  --int-steps 0
  --input-shape 224 224 96
  --lesion-loss-mode l2
  --lesion-sigma-vox 2.0
  --lesion-beta 2.0
  --phase2-epoch 50
  --sim-weight2 0.15
  --reg-weight2 0.05
  --lesion-sigma-vox2 1.5
  --lesion-beta2 4.0
  --skip-missing-lesion
  --early-stop-metric lesion_err
  --early-stop-patience 30
  --early-stop-min-delta 0.001
  --save-every 10
)

# ---- Sweep-Varianten (Array 0..5) ----
# Fokus: bessere z-Performance & stabileres Fine-Tuning
# Du bekommst pro Task ein eigenes exp-id + eigene axis weights + lesion weights.
case "${SLURM_ARRAY_TASK_ID}" in
  0)
    # dein aktuelles Beispiel (als Referenz im Sweep)
    EXP="LES_MM_L2_2PH_aw1_1_0p5_ref"
    EXTRA=( --lesion-w-mm 1 1 0.5 --lesion-weight 0.8 --lesion-weight2 2.0 --exp-id "$EXP" )
    ;;
  1)
    # isotrop, aber gleiche Gewichte -> fairer Vergleich zu axisw
    EXP="LES_MM_L2_2PH_aw1_1_1"
    EXTRA=( --lesion-w-mm 1 1 1 --lesion-weight 0.8 --lesion-weight2 2.0 --exp-id "$EXP" )
    ;;
  2)
    # z stärker gewichten (typisch sinnvoll, weil sz groß)
    EXP="LES_MM_L2_2PH_aw1_1_2"
    EXTRA=( --lesion-w-mm 1 1 2 --lesion-weight 0.8 --lesion-weight2 2.0 --exp-id "$EXP" )
    ;;
  3)
    # z noch stärker, dafür Phase2 nicht zu aggressiv lassen (gleich wie base)
    EXP="LES_MM_L2_2PH_aw1_1_3"
    EXTRA=( --lesion-w-mm 1 1 3 --lesion-weight 0.8 --lesion-weight2 2.0 --exp-id "$EXP" )
    ;;
  4)
    # früherer Phase2 + etwas stärkeres lesion_weight2 (mehr Landmark-Druck)
    EXP="LES_MM_L2_2PH_aw1_1_2_p2e35_lw2p5"
    EXTRA=( --lesion-w-mm 1 1 2 --lesion-weight 0.8 --phase2-epoch 35 --lesion-weight2 2.5 --exp-id "$EXP" )
    ;;
  5)
    # stabiler “safe” run: etwas kleinere lesion weights (weniger Overfit), später Phase2
    EXP="LES_MM_L2_2PH_aw1_1_2_safe"
    EXTRA=( --lesion-w-mm 1 1 2 --lesion-weight 0.6 --phase2-epoch 60 --lesion-weight2 1.6 --exp-id "$EXP" )
    ;;
  *)
    echo "Unknown SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
    exit 1
    ;;
esac

echo "EXP: $EXP"
echo "CMD: $PY $SCRIPT ${BASE_ARGS[*]} ${EXTRA[*]}"

"$PY" "$SCRIPT" "${BASE_ARGS[@]}" "${EXTRA[@]}"

STATUS=$?
echo "Ende:  $(date)"
echo "Exit:  $STATUS"
exit $STATUS
