# Breast Biopsy Registration — Masterarbeit

Code zur deformierbaren Registrierung von Brust-MRT für die Biopsieplanung.
Ziel ist die genaue Lokalisation von **Läsion** und **Nadelspitze** zwischen
prä- und intra-interventionellen Aufnahmen. Verglichen werden drei
Registrierungs-Ansätze:

- **GMA-RAFT-3D** — optical-flow-basierte 3D-Registrierung (RAFT mit Global Motion Aggregation)
- **TransMorph** — Transformer-basiertes Registrierungsnetz
- **VoxelMorph** — CNN-basiertes unsupervised Registrierungsnetz

Beide Aufgaben (Läsion, Nadelspitze) werden mit mehreren Ähnlichkeits-Losses
untersucht: **MSE / L2**, **NCC** und **MI**, jeweils auch in einer **axialen**
Variante und einer **2-Phasen**-Variante mit Multi-Modal-Heatmap.

## Struktur

| Ordner | Inhalt |
|--------|--------|
| [`GMARAFT_3d/`](GMARAFT_3d/) | GMA-RAFT-3D Codebasis: Netzwerk (`network/`, `network_3d/`), Training (`train/`, `train2/`), Daten-Loader (`loader/`), Configs (`configs/`), Eval (`eval*.py`, `batch_eval.py`), Inferenz/Registrierung (`infer_*.py`, `register_*.py`, `run_pairwise.py`) |
| [`transmorph/`](transmorph/) | TransMorph-Trainingsskripte: Biopsie (`train_transmorph_biopsy.py`) und axiale Variante (`..._axial.py`) |
| [`voxelmorph/`](voxelmorph/) | VoxelMorph-Trainingsskripte: Läsion (`vxm_train_lesion.py`), 2-Phasen MSE/L2 (`vxm_lesion_mm_l2_2phase*.py`), axiale Heatmap-Sweep-Variante (`train_vxm_lesion_axis_mse_mmheatmap_sweep.py`) |
| [`notebooks/`](notebooks/) | Auswertung & Doku-Plots: Distanz-Metriken GT↔Prediction (Läsion/Nadelspitze), Pipeline-Schwellwerte, Doku-Abbildungen |
| [`sbatches/`](sbatches/) | SLURM-Job- (`jobs/`) und W&B-Sweep-Skripte (`sweeps/`), nach Modell gegliedert: `gmaraft3d/`, `transmorph/`, `vxm/` (letzteres weiter nach Loss `mse`/`ncc`/`mi` und Aufgabe `lesion`/`needletip`) |

## Umgebung

Python 3.10, PyTorch auf GPU, Conda-Umgebung `biopsy_env`.

```bash
conda env create -f environment.yml      # erstellt das Env biopsy_env
conda activate biopsy_env
# oder reine pip-Abhängigkeiten:
pip install -r requirements_frozen.txt
```

- [`environment.yml`](environment.yml) — vollständige Conda-Spezifikation.
- [`requirements_frozen.txt`](requirements_frozen.txt) — eingefrorene pip-Abhängigkeiten.

## Training & Sweeps (SLURM)

Training läuft auf einem SLURM-Cluster mit GPU. Job- und Sweep-Skripte liegen
gebündelt unter [`sbatches/`](sbatches/):

```bash
# Einzeljob abschicken
sbatch sbatches/vxm/ncc/lesion/jobs/vxm_lesion_250.sbatch

# W&B-Sweep starten
wandb sweep sbatches/vxm/ncc/lesion/sweeps/sweep_ncc.yaml
```

## Hinweise

- **Patientendaten** (`data/`, `*.nii`, `*.nii.gz`, `*.dcm`) werden niemals committet — siehe [`.gitignore`](.gitignore).
- Modell-Checkpoints (`*.pth`, `*.ckpt`, `*.h5`) und Trainings-Logs (`wandb/`, `runs/`) sind ebenfalls ausgeschlossen.
- Die Pfade in den `.sbatch`-Skripten (Conda-Env, Logs, Daten) verweisen auf das Cluster-Setup und müssen für andere Umgebungen angepasst werden.
