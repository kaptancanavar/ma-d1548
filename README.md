# Breast Biopsy Registration — Masterarbeit

Code zur deformierbaren Registrierung von Brust-MRT für die Biopsieplanung
(Läsions- und Nadelspitzen-Lokalisation). Verglichen werden drei Registrierungs-Ansätze:
GMA-RAFT-3D, TransMorph und VoxelMorph.

## Struktur

| Ordner | Inhalt |
|--------|--------|
| [`GMARAFT_3d/`](GMARAFT_3d/) | GMA-RAFT-3D Codebasis: Netzwerk (`network/`, `network_3d/`), Training (`train/`, `train2/`), Daten-Loader (`loader/`), Configs, Eval-/Inferenz-/Registrierungs-Skripte |
| [`transmorph/`](transmorph/) | TransMorph-Trainingsskripte (Biopsie, axial) |
| [`voxelmorph/`](voxelmorph/) | VoxelMorph-Trainingsskripte (Läsion, 2-Phasen-MSE/L2, Heatmap-Sweep) |
| [`notebooks/`](notebooks/) | Auswertung & Doku-Plots (Distanz-Metriken GT↔Prediction, Pipeline-Schwellwerte) |

## Hinweise

- **Patientendaten** (`data/`, `*.nii`, `*.nii.gz`, `*.dcm`) werden niemals committet — siehe [`.gitignore`](.gitignore).
- Modell-Checkpoints (`*.pth`, `*.ckpt`, `*.h5`) und Trainings-Logs (`wandb/`, `runs/`) sind ebenfalls ausgeschlossen.
