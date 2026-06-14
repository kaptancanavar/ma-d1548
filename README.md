# Breast Biopsy Registration — Master's Thesis

Code for deformable registration of breast MRI for biopsy planning.
The goal is the accurate localization of the **lesion** and the **needle tip**
between pre- and intra-interventional scans. Three registration approaches are
compared:

- **GMA-RAFT-3D** — optical-flow-based 3D registration (RAFT with Global Motion Aggregation)
- **TransMorph** — Transformer-based registration network
- **VoxelMorph** — CNN-based unsupervised registration network

Both tasks (lesion, needle tip) are studied with several similarity losses:
**MSE / L2**, **NCC** and **MI**, each also in an **axial** variant and a
**two-phase** variant with a multi-modal heatmap.

## Structure

| Folder | Contents |
|--------|----------|
| [`GMARAFT_3d/`](GMARAFT_3d/) | GMA-RAFT-3D codebase: network (`network/`, `network_3d/`), training (`train/`, `train2/`), data loaders (`loader/`), configs (`configs/`), evaluation (`eval*.py`, `batch_eval.py`), inference/registration (`infer_*.py`, `register_*.py`, `run_pairwise.py`) |
| [`transmorph/`](transmorph/) | TransMorph training scripts: biopsy (`train_transmorph_biopsy.py`) and axial variant (`..._axial.py`) |
| [`voxelmorph/`](voxelmorph/) | VoxelMorph training scripts: lesion (`vxm_train_lesion.py`), two-phase MSE/L2 (`vxm_lesion_mm_l2_2phase*.py`), axial heatmap-sweep variant (`train_vxm_lesion_axis_mse_mmheatmap_sweep.py`) |
| [`notebooks/`](notebooks/) | Analysis & documentation plots: distance metrics GT↔prediction (lesion/needle tip), pipeline thresholds, documentation figures |
| [`sbatches/`](sbatches/) | SLURM job (`jobs/`) and W&B sweep scripts (`sweeps/`), organized by model: `gmaraft3d/`, `transmorph/`, `vxm/` (the latter further split by loss `mse`/`ncc`/`mi` and task `lesion`/`needletip`) |

## Environment

Python 3.10, PyTorch on GPU, Conda environment `biopsy_env`.

```bash
conda env create -f environment.yml      # creates the biopsy_env environment
conda activate biopsy_env
# or plain pip dependencies:
pip install -r requirements_frozen.txt
```

- [`environment.yml`](environment.yml) — full Conda specification.
- [`requirements_frozen.txt`](requirements_frozen.txt) — frozen pip dependencies.

## Training & Sweeps (SLURM)

Training runs on a SLURM cluster with GPU. Job and sweep scripts are bundled
under [`sbatches/`](sbatches/):

```bash
# submit a single job
sbatch sbatches/vxm/ncc/lesion/jobs/vxm_lesion_250.sbatch

# start a W&B sweep
wandb sweep sbatches/vxm/ncc/lesion/sweeps/sweep_ncc.yaml
```

## Notes

- **Patient data** (`data/`, `*.nii`, `*.nii.gz`, `*.dcm`) is never committed — see [`.gitignore`](.gitignore).
- Model checkpoints (`*.pth`, `*.ckpt`, `*.h5`) and training logs (`wandb/`, `runs/`) are excluded as well.
- The paths in the `.sbatch` scripts (Conda env, logs, data) refer to the cluster setup and must be adjusted for other environments.
