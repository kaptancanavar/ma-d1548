#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GMARAFT3D sweep training (stable, NaN-robust) for breast biopsy registration.

Enhancements vs. your current script:
- Similarity loss options:
  - global_ncc (FP32), local_ncc (FP32), mse, mixed (alpha*NCC + (1-alpha)*MSE)
- Regularization options (all FP32):
  - 1st-derivative smoothness (L1)
  - bending energy (2nd-derivative)
  - optional Jacobian negative-determinant penalty (anti-folding)
  - optional flow magnitude penalty
- Keeps your stability guards:
  - inputs: z-score -> clamp -> [0,1] for GMARAFT
  - similarity computed on z-score volumes (FP32)
  - flow clamped before warping
  - fallback to MSE if sim becomes non-finite
  - abort if total loss non-finite

CSV expected columns: moving,fixed (optionally other metadata columns are ignored)
"""

import os
import csv
import time
import json
import argparse
import difflib
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Tuple, List

import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import wandb

from network_3d.model import GMARAFT_Denoiser3D


# ----------------------------- Config ----------------------------- #

@dataclass
class TrainCfg:
    train_csv: str
    val_csv: str
    out_root: str = "experiments_gmaraft"

    epochs: int = 50
    batch_size: int = 1
    num_workers: int = 4
    lr: float = 2e-4
    weight_decay: float = 5e-5

    # Shapes are given as (X, Y, Z)
    input_shape: Tuple[int, int, int] = (224, 224, 96)
    train_shape: Tuple[int, int, int] = (96, 96, 64)

    # Similarity loss config
    sim_type: str = "local_ncc"  # global_ncc | local_ncc | mse | mixed
    ncc_win: int = 9             # for local_ncc
    mixed_alpha: float = 0.8     # for mixed: alpha*NCC + (1-alpha)*MSE

    # Loss weights
    sim_weight: float = 1.0
    reg_weight: float = 0.05     # overall reg weight (applied to smooth+bend mix)
    smooth_frac: float = 0.5     # fraction of reg_weight on 1st-derivative smoothness
    bend_frac: float = 0.5       # fraction of reg_weight on bending energy

    jac_weight: float = 0.0      # anti-folding penalty weight (try 1e-3..1e-2)
    mag_weight: float = 0.0      # flow magnitude penalty (try 1e-4..1e-3)

    # optional landmark supervision
    # - source "auto": prefer CSV columns mv_lm_*_mm/fx_lm_*_mm; otherwise try MITK .ano.json in an `annotations/` folder near the volumes
    # - source "csv": only CSV columns
    # - source "json": only JSON auto-discovery
    # - source "none": disable even if lm_weight>0
    lm_weight: float = 0.0       # try 0.1..1.0
    lm_iters: int = 8            # fixed-point iterations for inverting fixed->moving flow
    lm_source: str = "auto"      # auto|csv|json|none
    lm_preset: str = "lesion"    # lesion|tip|custom
    lm_names: str = ""           # comma-separated point names (overrides preset if non-empty)
    anno_dir_name: str = "annotations"
    anno_max_up: int = 8

    # multi-step weighting
    gamma: float = 0.85

    # stability
    zclamp: float = 5.0
    flow_clamp: float = 20.0
    grad_clip: float = 1.0
    use_amp: bool = True

    save_every: int = 10
    seed: int = 42

    # wandb
    wandb_project: str = "breast-biopsy-gmaraft3d"
    wandb_group: str = "gmaraft_sweep"
    wandb_name: str = ""
    wandb_mode: str = "online"  # online|offline|disabled


def parse_args() -> TrainCfg:
    p = argparse.ArgumentParser("GMARAFT3D sweep training (stable + flexible losses)")

    p.add_argument("--train-csv", type=str, required=True)
    p.add_argument("--val-csv", type=str, required=True)
    p.add_argument("--out-root", type=str, default="experiments_gmaraft")

    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=5e-5)

    p.add_argument("--input-shape", type=int, nargs=3, default=[224, 224, 96])
    p.add_argument("--train-shape", type=int, nargs=3, default=[96, 96, 64])

    # similarity options
    p.add_argument("--sim-type", type=str, default="local_ncc",
                   choices=["global_ncc", "local_ncc", "mse", "mixed"])
    p.add_argument("--ncc-win", type=int, default=9)
    p.add_argument("--mixed-alpha", type=float, default=0.8)

    # weights
    p.add_argument("--sim-weight", type=float, default=1.0)
    p.add_argument("--reg-weight", type=float, default=0.05)
    p.add_argument("--smooth-frac", type=float, default=0.5)
    p.add_argument("--bend-frac", type=float, default=0.5)
    p.add_argument("--jac-weight", type=float, default=0.0)
    p.add_argument("--mag-weight", type=float, default=0.0)

    # landmark supervision (optional)
    p.add_argument("--lm-weight", type=float, default=0.0)
    p.add_argument("--lm-iters", type=int, default=8)
    p.add_argument("--lm-source", type=str, default="auto", choices=["auto", "csv", "json", "none"],
               help="Where to read landmarks from: CSV columns, MITK .ano.json auto-discovery, or auto.")
    p.add_argument("--lm-preset", type=str, default="lesion", choices=["lesion", "tip", "custom"],
               help="Convenience preset for common point names. Use 'custom' with --lm-names.")
    p.add_argument("--lm-names", type=str, default="",
               help="Comma-separated point names to accept (case-insensitive). Overrides --lm-preset if non-empty.")
    p.add_argument("--anno-dir-name", type=str, default="annotations",
               help="Annotation directory name to search for when lm-source is json/auto.")
    p.add_argument("--anno-max-up", type=int, default=8,
               help="How many parent directories to search for anno-dir-name.")

    p.add_argument("--gamma", type=float, default=0.85)

    p.add_argument("--zclamp", type=float, default=5.0)
    p.add_argument("--flow-clamp", type=float, default=20.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--use-amp", action="store_true")
    p.add_argument("--no-amp", dest="use_amp", action="store_false")
    p.set_defaults(use_amp=True)

    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--wandb-project", type=str, default="breast-biopsy-gmaraft3d")
    p.add_argument("--wandb-group", type=str, default="gmaraft_sweep")
    p.add_argument("--wandb-name", type=str, default="")
    p.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])

    a = p.parse_args()
    return TrainCfg(
        train_csv=a.train_csv,
        val_csv=a.val_csv,
        out_root=a.out_root,
        epochs=a.epochs,
        batch_size=a.batch_size,
        num_workers=a.num_workers,
        lr=a.lr,
        weight_decay=a.weight_decay,
        input_shape=tuple(a.input_shape),
        train_shape=tuple(a.train_shape),
        sim_type=a.sim_type,
        ncc_win=a.ncc_win,
        mixed_alpha=a.mixed_alpha,
        sim_weight=a.sim_weight,
        reg_weight=a.reg_weight,
        smooth_frac=a.smooth_frac,
        bend_frac=a.bend_frac,
        jac_weight=a.jac_weight,
        mag_weight=a.mag_weight,
        lm_weight=a.lm_weight,
        lm_iters=a.lm_iters,
        lm_source=a.lm_source,
        lm_preset=a.lm_preset,
        lm_names=a.lm_names,
        anno_dir_name=a.anno_dir_name,
        anno_max_up=a.anno_max_up,
        gamma=a.gamma,
        zclamp=a.zclamp,
        flow_clamp=a.flow_clamp,
        grad_clip=a.grad_clip,
        use_amp=a.use_amp,
        save_every=a.save_every,
        seed=a.seed,
        wandb_project=a.wandb_project,
        wandb_group=a.wandb_group,
        wandb_name=a.wandb_name,
        wandb_mode=a.wandb_mode,
    )


# ----------------------------- Dataset ----------------------------- #

class BreastPairDataset(Dataset):
    """
    Returns:
      fixed_in, moving_in, fixed_z, moving_z, fixed_pt_xyz, moving_pt_xyz, has_lm

    Volumes are loaded as XYZ (X,Y,Z) from NIfTI, then converted to DHW (Z,Y,X) for PyTorch.

    - *_in: z-score -> clamp -> mapped to [0,1] (stable inputs for GMARAFT)
    - *_z : z-score volumes (used for similarity loss)

    Landmarks (optional):
      (A) CSV columns in mm/world:
          mv_lm_x_mm,mv_lm_y_mm,mv_lm_z_mm, fx_lm_x_mm,fx_lm_y_mm,fx_lm_z_mm
      (B) MITK .ano.json auto-discovery:
          finds an `{anno_dir_name}/` folder near each volume, picks the best matching JSON by filename similarity,
          then selects the first point matching lm_names (substring match; case-insensitive).
    """

    def __init__(
        self,
        csv_path: Path,
        input_shape_xyz: Tuple[int, int, int],
        train_shape_xyz: Tuple[int, int, int],
        zclamp: float,
        lm_source: str = "auto",
        lm_names: List[str] | None = None,
        anno_dir_name: str = "annotations",
        anno_max_up: int = 8,
    ):
        super().__init__()
        self.csv_path = Path(csv_path)
        self.input_shape_xyz = tuple(input_shape_xyz)
        self.train_shape_xyz = tuple(train_shape_xyz)
        self.zclamp = float(zclamp)

        self.lm_source = str(lm_source).lower()
        self.lm_names = [str(x).lower() for x in (lm_names or [])]
        self.anno_dir_name = str(anno_dir_name)
        self.anno_max_up = int(anno_max_up)

        if not self.csv_path.is_file():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

        self.rows: List[dict] = []
        with self.csv_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("moving") and row.get("fixed"):
                    self.rows.append(row)

        if not self.rows:
            raise RuntimeError(f"No valid rows in {self.csv_path}")

    def __len__(self) -> int:
        return len(self.rows)

    # ---------- IO / Geometry ----------

    @staticmethod
    def _load_nifti_xyz_affine(path: str) -> Tuple[np.ndarray, np.ndarray]:
        img = nib.load(path)
        vol = img.get_fdata().astype(np.float32)
        aff = np.array(img.affine, dtype=np.float64)
        vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)
        # NOTE: keep this mild clip, same as original script
        vol = np.clip(vol, -10.0, 10.0)
        if vol.ndim != 3:
            raise ValueError(f"Expected 3D volume, got {vol.shape} at {path}")
        return vol, aff

    @staticmethod
    def _mm_to_vox_xyz(pt_mm: np.ndarray, affine: np.ndarray) -> np.ndarray:
        """Convert world/mm coords to voxel coords (x,y,z) in the loaded XYZ index space."""
        pt_mm = np.asarray(pt_mm, dtype=np.float64).reshape(3)
        inv = np.linalg.inv(affine)
        pt_vox = nib.affines.apply_affine(inv, pt_mm)
        return np.asarray(pt_vox, dtype=np.float32)

    @staticmethod
    def _map_point_to_resized_grid_align_corners_false(
        pt_xyz: np.ndarray,
        in_xyz: Tuple[int, int, int],
        out_xyz: Tuple[int, int, int],
    ) -> np.ndarray:
        """Map voxel coords from an input grid to an output grid as used by F.interpolate(..., align_corners=False)."""
        pt_xyz = np.asarray(pt_xyz, dtype=np.float32).reshape(3)
        in_x, in_y, in_z = map(float, in_xyz)
        out_x, out_y, out_z = map(float, out_xyz)
        x = (pt_xyz[0] + 0.5) * (out_x / in_x) - 0.5
        y = (pt_xyz[1] + 0.5) * (out_y / in_y) - 0.5
        z = (pt_xyz[2] + 0.5) * (out_z / in_z) - 0.5
        return np.array([x, y, z], dtype=np.float32)

    # ---------- Annotation discovery (MITK json) ----------

    @staticmethod
    def _strip_nii_suffix(p: Path) -> str:
        name = p.name
        if name.endswith(".nii.gz"):
            return name[:-7]
        if name.endswith(".nii"):
            return name[:-4]
        return p.stem

    @staticmethod
    def _norm_name(s: str) -> str:
        """Normalize a point name for robust matching (case-insensitive, ignore non-alphanumerics)."""
        return "".join(ch for ch in str(s).lower() if ch.isalnum())

    def _find_annotations_dir(self, vol_path: str) -> Path:
        cur = Path(vol_path).resolve().parent
        for _ in range(max(0, self.anno_max_up) + 1):
            cand = cur / self.anno_dir_name
            if cand.is_dir():
                return cand
            cur = cur.parent
        raise FileNotFoundError(
            f"Could not find '{self.anno_dir_name}/' within {self.anno_max_up} parents of {vol_path}"
        )

    def _best_matching_json(self, anno_dir: Path, vol_path: str) -> Path:
        vol_stem = self._strip_nii_suffix(Path(vol_path)).lower()
        jsons = sorted(anno_dir.glob("*.json"))
        if not jsons:
            raise FileNotFoundError(f"No .json files found in {anno_dir}")
        best = None
        best_score = -1.0
        for jp in jsons:
            js = jp.stem.lower()
            score = difflib.SequenceMatcher(a=vol_stem, b=js).ratio()
            if score > best_score:
                best_score = score
                best = jp
        if best is None:
            raise FileNotFoundError(f"No matching JSON found in {anno_dir} for {vol_path}")
        return best

    @staticmethod
    def _load_mitk_points_mm(json_path: Path) -> List[Tuple[str, np.ndarray]]:
        """
        Returns list of (name_lower, xyz_mm) from a MITK-style .ano.json.
        Tries 'coords_transformed' first (mm/world), otherwise falls back to 'coords'.
        """
        data = json.loads(json_path.read_text())
        out: List[Tuple[str, np.ndarray]] = []
        for ann in data.get("annotations", []):
            for p in ann.get("points", []):
                name = p.get("name") or p.get("label") or p.get("id") or ""
                name_l = str(name).lower()
                xyz = p.get("coords_transformed") or p.get("coords")
                if xyz is None:
                    continue
                if not isinstance(xyz, (list, tuple)) or len(xyz) < 3:
                    continue
                out.append((name_l, np.array([float(xyz[0]), float(xyz[1]), float(xyz[2])], dtype=np.float32)))
        return out

    def _pick_point_mm(self, pts: List[Tuple[str, np.ndarray]]) -> np.ndarray:
        if not pts:
            raise ValueError("No points in JSON.")
        if not self.lm_names:
            # if no names provided: just take the first point
            return pts[0][1]

        # normalized keys (e.g. "needle-tip" -> "needletip")
        keys_raw = [k for k in self.lm_names if k]
        keys_norm = [self._norm_name(k) for k in keys_raw]

        # 1) Prefer exact normalized match
        for name_l, xyz in pts:
            n_norm = self._norm_name(name_l)
            for k_norm in keys_norm:
                if k_norm and n_norm == k_norm:
                    return xyz

        # 2) Fallback: substring match (useful for "lesion_1", "NeedleTip (auto)", etc.)
        for name_l, xyz in pts:
            n_norm = self._norm_name(name_l)
            for k_raw, k_norm in zip(keys_raw, keys_norm):
                if (k_raw and k_raw in name_l) or (k_norm and k_norm in n_norm):
                    return xyz

        raise ValueError(f"No point matched lm_names={self.lm_names}; available={[n for n, _ in pts]}")

    def _get_landmark_mm_from_json(self, vol_path: str) -> np.ndarray:
        anno_dir = self._find_annotations_dir(vol_path)
        js = self._best_matching_json(anno_dir, vol_path)
        pts = self._load_mitk_points_mm(js)
        return self._pick_point_mm(pts)

    # ---------- Preprocessing ----------

    @staticmethod
    def _zscore(vol: np.ndarray) -> np.ndarray:
        m = float(vol.mean())
        s = float(vol.std())
        if s < 1e-5:
            s = 1e-5
        return ((vol - m) / s).astype(np.float32)

    @staticmethod
    def _xyz_to_dhw(vol_xyz: np.ndarray) -> np.ndarray:
        # (X,Y,Z) -> (D,H,W) = (Z,Y,X)
        return np.transpose(vol_xyz, (2, 1, 0)).copy()

    @staticmethod
    def _resample_1dhw(x_1dhw: torch.Tensor, target_dhw: Tuple[int, int, int]) -> torch.Tensor:
        # x_1dhw: (1,D,H,W) -> (1,1,D,H,W) -> interpolate -> (1,D,H,W)
        x = x_1dhw.unsqueeze(0)
        y = F.interpolate(x, size=target_dhw, mode="trilinear", align_corners=False)
        return y.squeeze(0)

    def _zscore_to_01(self, vol_z: np.ndarray) -> np.ndarray:
        c = self.zclamp
        v = np.clip(vol_z, -c, c)
        v = (v + c) / (2.0 * c)
        return v.astype(np.float32)

    # ---------- Item ----------

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        mv_path = row["moving"]
        fx_path = row["fixed"]

        mv_xyz, mv_aff = self._load_nifti_xyz_affine(mv_path)
        fx_xyz, fx_aff = self._load_nifti_xyz_affine(fx_path)

        if mv_xyz.shape != self.input_shape_xyz:
            raise ValueError(f"Moving shape {mv_xyz.shape} != expected {self.input_shape_xyz} at {mv_path}")
        if fx_xyz.shape != self.input_shape_xyz:
            raise ValueError(f"Fixed shape {fx_xyz.shape} != expected {self.input_shape_xyz} at {fx_path}")

        # z-score for similarity loss
        mv_z_xyz = self._zscore(mv_xyz)
        fx_z_xyz = self._zscore(fx_xyz)

        # inputs for the network in [0,1]
        mv_in_xyz = self._zscore_to_01(mv_z_xyz)
        fx_in_xyz = self._zscore_to_01(fx_z_xyz)

        # xyz -> dhw
        mv_z_dhw = self._xyz_to_dhw(mv_z_xyz)
        fx_z_dhw = self._xyz_to_dhw(fx_z_xyz)
        mv_in_dhw = self._xyz_to_dhw(mv_in_xyz)
        fx_in_dhw = self._xyz_to_dhw(fx_in_xyz)

        # torch (1,D,H,W)
        mv_z_t = torch.from_numpy(mv_z_dhw).unsqueeze(0)
        fx_z_t = torch.from_numpy(fx_z_dhw).unsqueeze(0)
        mv_in_t = torch.from_numpy(mv_in_dhw).unsqueeze(0)
        fx_in_t = torch.from_numpy(fx_in_dhw).unsqueeze(0)

        # resize to train shape (if used)
        if self.train_shape_xyz != self.input_shape_xyz:
            train_dhw = (self.train_shape_xyz[2], self.train_shape_xyz[1], self.train_shape_xyz[0])
            mv_in_t = self._resample_1dhw(mv_in_t, train_dhw)
            fx_in_t = self._resample_1dhw(fx_in_t, train_dhw)
            mv_z_t = self._resample_1dhw(mv_z_t, train_dhw)
            fx_z_t = self._resample_1dhw(fx_z_t, train_dhw)

        # -------- Landmarks (optional) --------
        has_lm = 0.0
        mv_pt = np.zeros((1, 3), dtype=np.float32)
        fx_pt = np.zeros((1, 3), dtype=np.float32)

        src = str(self.lm_source).lower()
        if src != "none":
            # (A) CSV columns
            if src in ("auto", "csv"):
                need_cols = [
                    "mv_lm_x_mm", "mv_lm_y_mm", "mv_lm_z_mm",
                    "fx_lm_x_mm", "fx_lm_y_mm", "fx_lm_z_mm",
                ]
                if all(k in row for k in need_cols):
                    try:
                        mv_mm = np.array(
                            [float(row["mv_lm_x_mm"]), float(row["mv_lm_y_mm"]), float(row["mv_lm_z_mm"])],
                            dtype=np.float64
                        )
                        fx_mm = np.array(
                            [float(row["fx_lm_x_mm"]), float(row["fx_lm_y_mm"]), float(row["fx_lm_z_mm"])],
                            dtype=np.float64
                        )
                        if np.isfinite(mv_mm).all() and np.isfinite(fx_mm).all():
                            mv_vox = self._mm_to_vox_xyz(mv_mm, mv_aff)
                            fx_vox = self._mm_to_vox_xyz(fx_mm, fx_aff)

                            if self.train_shape_xyz != self.input_shape_xyz:
                                mv_vox = self._map_point_to_resized_grid_align_corners_false(
                                    mv_vox, self.input_shape_xyz, self.train_shape_xyz
                                )
                                fx_vox = self._map_point_to_resized_grid_align_corners_false(
                                    fx_vox, self.input_shape_xyz, self.train_shape_xyz
                                )

                            mv_pt[0] = mv_vox
                            fx_pt[0] = fx_vox
                            has_lm = 1.0
                    except Exception:
                        has_lm = 0.0

            # (B) JSON auto-discovery
            if has_lm < 0.5 and src in ("auto", "json"):
                try:
                    mv_mm = self._get_landmark_mm_from_json(mv_path)
                    fx_mm = self._get_landmark_mm_from_json(fx_path)
                    if np.isfinite(mv_mm).all() and np.isfinite(fx_mm).all():
                        mv_vox = self._mm_to_vox_xyz(mv_mm, mv_aff)
                        fx_vox = self._mm_to_vox_xyz(fx_mm, fx_aff)

                        if self.train_shape_xyz != self.input_shape_xyz:
                            mv_vox = self._map_point_to_resized_grid_align_corners_false(
                                mv_vox, self.input_shape_xyz, self.train_shape_xyz
                            )
                            fx_vox = self._map_point_to_resized_grid_align_corners_false(
                                fx_vox, self.input_shape_xyz, self.train_shape_xyz
                            )

                        mv_pt[0] = mv_vox
                        fx_pt[0] = fx_vox
                        has_lm = 1.0
                except Exception:
                    has_lm = 0.0

        mv_pt_t = torch.from_numpy(mv_pt)  # (1,3)
        fx_pt_t = torch.from_numpy(fx_pt)  # (1,3)
        has_lm_t = torch.tensor(has_lm, dtype=torch.float32)

        return fx_in_t, mv_in_t, fx_z_t, mv_z_t, fx_pt_t, mv_pt_t, has_lm_t


# ----------------------------- Loss / Warp ----------------------------- #

class GlobalNCCLoss(nn.Module):
    """Global NCC as loss, computed in FP32."""
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x,y: (B,1,D,H,W)
        x = x[:, 0].reshape(x.shape[0], -1).float()
        y = y[:, 0].reshape(y.shape[0], -1).float()

        x = x - x.mean(dim=1, keepdim=True)
        y = y - y.mean(dim=1, keepdim=True)

        varx = (x * x).mean(dim=1, keepdim=True)
        vary = (y * y).mean(dim=1, keepdim=True)
        denom = torch.sqrt(varx * vary).clamp_min(self.eps)

        ncc = (x * y).mean(dim=1, keepdim=True) / denom
        return -ncc.mean()


class LocalNCCLoss(nn.Module):
    """Local (windowed) NCC for 3D, computed in FP32. Returns negative NCC."""
    def __init__(self, win: int = 9, eps: float = 1e-5):
        super().__init__()
        self.win = int(win)
        self.eps = eps

    def forward(self, I: torch.Tensor, J: torch.Tensor) -> torch.Tensor:
        # I,J: (B,1,D,H,W)
        I = I.float()
        J = J.float()

        win = self.win
        if win < 3:
            win = 3
        if win % 2 == 0:
            win += 1
        pad = win // 2

        filt = torch.ones((1, 1, win, win, win), device=I.device, dtype=I.dtype)

        def conv(x):
            return F.conv3d(x, filt, padding=pad)

        I2 = I * I
        J2 = J * J
        IJ = I * J

        I_sum  = conv(I)
        J_sum  = conv(J)
        I2_sum = conv(I2)
        J2_sum = conv(J2)
        IJ_sum = conv(IJ)

        win_size = float(win ** 3)

        I_mean = I_sum / win_size
        J_mean = J_sum / win_size

        cross = IJ_sum - J_mean * I_sum - I_mean * J_sum + I_mean * J_mean * win_size
        I_var = I2_sum - 2 * I_mean * I_sum + I_mean * I_mean * win_size
        J_var = J2_sum - 2 * J_mean * J_sum + J_mean * J_mean * win_size

        denom = (I_var * J_var).clamp_min(self.eps)
        ncc = (cross * cross) / denom
        return -ncc.mean()


def flow_smoothness_l1(flow: torch.Tensor) -> torch.Tensor:
    # flow: (B,3,D,H,W) - FP32
    flow = flow.float()
    B, C, D, H, W = flow.shape
    if D < 2 or H < 2 or W < 2:
        return flow.new_tensor(0.0)
    dx = (flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1]).abs().mean()
    dy = (flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :]).abs().mean()
    dz = (flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :]).abs().mean()
    return (dx + dy + dz) / 3.0


def bending_energy(flow: torch.Tensor) -> torch.Tensor:
    # 2nd derivatives - FP32
    f = flow.float()
    B, C, D, H, W = f.shape
    if D < 3 or H < 3 or W < 3:
        return f.new_tensor(0.0)

    ddx = (f[:, :, :, :, 2:] - 2.0 * f[:, :, :, :, 1:-1] + f[:, :, :, :, :-2]).abs().mean()
    ddy = (f[:, :, :, 2:, :] - 2.0 * f[:, :, :, 1:-1, :] + f[:, :, :, :-2, :]).abs().mean()
    ddz = (f[:, :, 2:, :, :] - 2.0 * f[:, :, 1:-1, :, :] + f[:, :, :-2, :, :]).abs().mean()
    return (ddx + ddy + ddz) / 3.0


def flow_magnitude_l2(flow: torch.Tensor) -> torch.Tensor:
    return (flow.float() ** 2).mean()


def jacobian_det_penalty(flow: torch.Tensor) -> torch.Tensor:
    """
    Penalize negative Jacobian determinant (folding) for transform x -> x + flow(x).
    flow: (B,3,D,H,W) in voxel units.
    """
    f = flow.float()
    B, C, D, H, W = f.shape
    if D < 3 or H < 3 or W < 3:
        return f.new_tensor(0.0)

    # Central differences (approx), then crop to common interior [1:-1]
    dfx_dx = (f[:, 0, :, :, 2:] - f[:, 0, :, :, :-2]) / 2.0
    dfx_dy = (f[:, 0, :, 2:, :] - f[:, 0, :, :-2, :]) / 2.0
    dfx_dz = (f[:, 0, 2:, :, :] - f[:, 0, :-2, :, :]) / 2.0

    dfy_dx = (f[:, 1, :, :, 2:] - f[:, 1, :, :, :-2]) / 2.0
    dfy_dy = (f[:, 1, :, 2:, :] - f[:, 1, :, :-2, :]) / 2.0
    dfy_dz = (f[:, 1, 2:, :, :] - f[:, 1, :-2, :, :]) / 2.0

    dfz_dx = (f[:, 2, :, :, 2:] - f[:, 2, :, :, :-2]) / 2.0
    dfz_dy = (f[:, 2, :, 2:, :] - f[:, 2, :, :-2, :]) / 2.0
    dfz_dz = (f[:, 2, 2:, :, :] - f[:, 2, :-2, :, :]) / 2.0

    # bring all to shape (B, D-2, H-2, W-2)
    dfx_dx = dfx_dx[:, 1:-1, 1:-1, :]
    dfx_dy = dfx_dy[:, 1:-1, :, 1:-1]
    dfx_dz = dfx_dz[:, :, 1:-1, 1:-1]

    dfy_dx = dfy_dx[:, 1:-1, 1:-1, :]
    dfy_dy = dfy_dy[:, 1:-1, :, 1:-1]
    dfy_dz = dfy_dz[:, :, 1:-1, 1:-1]

    dfz_dx = dfz_dx[:, 1:-1, 1:-1, :]
    dfz_dy = dfz_dy[:, 1:-1, :, 1:-1]
    dfz_dz = dfz_dz[:, :, 1:-1, 1:-1]

    J11 = 1.0 + dfx_dx
    J12 = dfx_dy
    J13 = dfx_dz
    J21 = dfy_dx
    J22 = 1.0 + dfy_dy
    J23 = dfy_dz
    J31 = dfz_dx
    J32 = dfz_dy
    J33 = 1.0 + dfz_dz

    detJ = (
        J11 * (J22 * J33 - J23 * J32)
        - J12 * (J21 * J33 - J23 * J31)
        + J13 * (J21 * J32 - J22 * J31)
    )

    return F.relu(-detJ).mean()


def warp_3d(moving: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """
    moving: (B,1,D,H,W)
    flow:   (B,3,D,H,W) with channels (dx,dy,dz) in voxel units (x=W,y=H,z=D)
    """
    moving = moving.float()
    flow = flow.float()

    B, C, D, H, W = moving.shape

    z = torch.linspace(0, D - 1, D, device=moving.device)
    y = torch.linspace(0, H - 1, H, device=moving.device)
    x = torch.linspace(0, W - 1, W, device=moving.device)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    base = torch.stack((xx, yy, zz), dim=-1)[None].repeat(B, 1, 1, 1, 1)  # (B,D,H,W,3)

    coords = base + flow.permute(0, 2, 3, 4, 1)  # (B,D,H,W,3)

    coords[..., 0] = 2.0 * coords[..., 0] / max(W - 1, 1) - 1.0
    coords[..., 1] = 2.0 * coords[..., 1] / max(H - 1, 1) - 1.0
    coords[..., 2] = 2.0 * coords[..., 2] / max(D - 1, 1) - 1.0

    return F.grid_sample(moving, coords, mode="bilinear",
                         padding_mode="border", align_corners=True)


def sample_flow_at_points(flow: torch.Tensor, pts_xyz: torch.Tensor) -> torch.Tensor:
    """
    flow: (B,3,D,H,W) fixed->moving (backward map), voxel units
    pts_xyz: (B,N,3) points in voxel coords (x=W, y=H, z=D)
    returns: (B,N,3) sampled flow at each point (dx,dy,dz)
    """
    flow = flow.float()
    B, _, D, H, W = flow.shape
    x = pts_xyz[..., 0]
    y = pts_xyz[..., 1]
    z = pts_xyz[..., 2]

    gx = 2.0 * x / max(W - 1, 1) - 1.0
    gy = 2.0 * y / max(H - 1, 1) - 1.0
    gz = 2.0 * z / max(D - 1, 1) - 1.0

    grid = torch.stack([gx, gy, gz], dim=-1)  # (B,N,3)
    grid = grid.view(B, -1, 1, 1, 3)

    samp = F.grid_sample(flow, grid, mode="bilinear", padding_mode="border", align_corners=True)
    # samp: (B,3,N,1,1) -> (B,N,3)
    return samp.squeeze(-1).squeeze(-1).permute(0, 2, 1)


def moving_points_to_fixed(flow: torch.Tensor, pts_moving_xyz: torch.Tensor, iters: int = 8) -> torch.Tensor:
    """
    Invert fixed->moving flow to map moving points into fixed space.

    Given backward map: x_m = x_f + flow(x_f)
    Solve for x_f given x_m via fixed-point iteration:
        x_f^{k+1} = x_m - flow(x_f^k)

    flow: (B,3,D,H,W)
    pts_moving_xyz: (B,N,3)
    returns: (B,N,3) predicted fixed points
    """
    flow = flow.float()
    B, _, D, H, W = flow.shape
    x = pts_moving_xyz.float().clone()
    for _ in range(int(iters)):
        f = sample_flow_at_points(flow, x)
        x = pts_moving_xyz.float() - f
        x[..., 0].clamp_(0, W - 1)
        x[..., 1].clamp_(0, H - 1)
        x[..., 2].clamp_(0, D - 1)
    return x


# ----------------------------- Helpers ----------------------------- #

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_run_dir(out_root: Path, run_name: str, run_id: str) -> Path:
    ts = datetime.now().strftime("%y%m%d_%H%M%S")
    safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in run_name)[:80]
    d = out_root / f"{ts}_{safe_name}_{run_id}"
    (d / "checkpoints").mkdir(parents=True, exist_ok=True)
    return d


def compute_similarity(
    fixed_z: torch.Tensor,
    warped_z: torch.Tensor,
    sim_type: str,
    global_ncc: nn.Module,
    local_ncc: nn.Module,
    mixed_alpha: float,
) -> torch.Tensor:
    """
    Returns sim loss (lower is better). Always FP32 inside.
    """
    sim_type = str(sim_type).lower()

    if sim_type == "mse":
        return F.mse_loss(fixed_z.float(), warped_z.float())

    if sim_type == "global_ncc":
        return global_ncc(fixed_z, warped_z)

    if sim_type == "local_ncc":
        return local_ncc(fixed_z, warped_z)

    if sim_type == "mixed":
        # NCC term depends on which NCC module is active (prefer local if provided)
        # We'll use local_ncc if it's set, else global.
        ncc_term = local_ncc(fixed_z, warped_z) if local_ncc is not None else global_ncc(fixed_z, warped_z)
        mse_term = F.mse_loss(fixed_z.float(), warped_z.float())
        a = float(mixed_alpha)
        a = max(0.0, min(1.0, a))
        return a * ncc_term + (1.0 - a) * mse_term

    # fallback
    return global_ncc(fixed_z, warped_z)


def resolve_lm_names(preset: str, names_csv: str) -> List[str]:
    """
    Return a list of accepted point-name keys.

    Matching in the dataset is case-insensitive and uses a *normalized* comparison:
    - exact normalized match first (e.g. "NeedleTip", "needle_tip", "needle-tip" -> "needletip")
    - then substring match as fallback (e.g. "lesion_1" contains "lesion")

    If you pass --lm-names, it overrides the preset.
    """
    preset = str(preset).lower().strip()
    names_csv = str(names_csv).strip()

    if names_csv:
        names = [n.strip().lower() for n in names_csv.split(",") if n.strip()]
        return names

    # Your dataset uses point names like "Lesion" or "Needletip"
    if preset == "lesion":
        return ["lesion"]
    if preset == "tip":
        return ["needletip", "needle tip", "needle-tip"]

    # custom/unknown -> empty => will pick first point in JSON (not recommended)
    return []

# ----------------------------- Main ----------------------------- #

def main():
    cfg = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. This training script expects a GPU node.")

    device = torch.device("cuda")
    set_seed(cfg.seed)
    torch.backends.cudnn.benchmark = True

    # W&B init (sweep overrides these values in wandb.config)
    os.environ["WANDB_MODE"] = cfg.wandb_mode
    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", cfg.wandb_project),
        group=os.environ.get("WANDB_GROUP", cfg.wandb_group),
        name=cfg.wandb_name if cfg.wandb_name else None,
        config=asdict(cfg),
    )
    wc = wandb.config  # source of truth

    out_root = Path(wc.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    run_name = run.name if run and run.name else "gmaraft3d"
    run_id = run.id if run else "no_wandb"
    exp_dir = make_run_dir(out_root, run_name, run_id)
    ckpt_dir = exp_dir / "checkpoints"

    # store config for reproducibility
    with (exp_dir / "config.json").open("w") as f:
        json.dump(dict(wc), f, indent=2)

    input_shape_xyz = tuple(wc.input_shape)
    train_shape_xyz = tuple(wc.train_shape)

    lm_names = resolve_lm_names(str(wc.lm_preset), str(wc.lm_names))

    train_ds = BreastPairDataset(Path(wc.train_csv), input_shape_xyz, train_shape_xyz,
                               zclamp=float(wc.zclamp),
                               lm_source=str(wc.lm_source), lm_names=lm_names,
                               anno_dir_name=str(wc.anno_dir_name), anno_max_up=int(wc.anno_max_up))
    val_ds = BreastPairDataset(Path(wc.val_csv), input_shape_xyz, train_shape_xyz,
                             zclamp=float(wc.zclamp),
                             lm_source=str(wc.lm_source), lm_names=lm_names,
                             anno_dir_name=str(wc.anno_dir_name), anno_max_up=int(wc.anno_max_up))

    train_loader = DataLoader(
        train_ds,
        batch_size=int(wc.batch_size),
        shuffle=True,
        num_workers=int(wc.num_workers),
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=max(1, int(wc.num_workers) // 2),
        pin_memory=True,
        drop_last=False,
    )

    model = GMARAFT_Denoiser3D().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(wc.lr),
        weight_decay=float(wc.weight_decay),
    )

    use_amp = bool(wc.use_amp)
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # similarity modules (FP32 internally)
    global_ncc = GlobalNCCLoss().to(device)
    local_ncc = LocalNCCLoss(win=int(wc.ncc_win)).to(device) if str(wc.sim_type).lower() in ["local_ncc", "mixed"] else None

    best_val = float("inf")

    # wandb summary
    wandb.run.summary["num_train_pairs"] = len(train_ds)
    wandb.run.summary["num_val_pairs"] = len(val_ds)
    wandb.run.summary["exp_dir"] = str(exp_dir)

    # normalize reg fractions
    smooth_frac = float(wc.smooth_frac)
    bend_frac = float(wc.bend_frac)
    s = smooth_frac + bend_frac
    if s <= 1e-8:
        smooth_frac, bend_frac = 1.0, 0.0
    else:
        smooth_frac /= s
        bend_frac /= s

    print("==== GMARAFT3D Sweep Training (stable + flexible losses) ====")
    print("Exp dir   :", exp_dir)
    print("Train     :", wc.train_csv, "pairs=", len(train_ds))
    print("Val       :", wc.val_csv, "pairs=", len(val_ds))
    print("Input xyz :", input_shape_xyz, " Train xyz:", train_shape_xyz)
    print("LR        :", float(wc.lr))
    print("Sim       :", str(wc.sim_type), " sim_weight:", float(wc.sim_weight), " ncc_win:", int(wc.ncc_win), " mixed_alpha:", float(wc.mixed_alpha))
    print("Reg       :", "reg_weight:", float(wc.reg_weight), f" smooth_frac:{smooth_frac:.2f} bend_frac:{bend_frac:.2f}",
          " jac_weight:", float(wc.jac_weight), " mag_weight:", float(wc.mag_weight))
    print("Landmarks :", "lm_weight:", float(wc.lm_weight), " lm_iters:", int(wc.lm_iters),
          " lm_source:", str(wc.lm_source), " lm_preset:", str(wc.lm_preset), " lm_names:", str(wc.lm_names),
          " anno_dir:", str(wc.anno_dir_name), " anno_max_up:", int(wc.anno_max_up))
    print("gamma     :", float(wc.gamma))
    print("zclamp    :", float(wc.zclamp), " flow_clamp:", float(wc.flow_clamp), " AMP:", use_amp)
    print("============================================================")

    for epoch in range(1, int(wc.epochs) + 1):
        t0 = time.time()

        # ----------------- TRAIN ----------------- #
        model.train()
        train_sum = 0.0
        train_batches = 0

        # component tracking (weighted sum across flow_preds)
        tr_sim_sum = 0.0
        tr_smooth_sum = 0.0
        tr_bend_sum = 0.0
        tr_jac_sum = 0.0
        tr_mag_sum = 0.0
        tr_lm_sum = 0.0
        tr_lm_valid_sum = 0.0  # sum(lm) over valid samples only
        tr_lm_cov = 0
        tr_total = 0

        for fixed_in, moving_in, fixed_z, moving_z, fixed_pt, moving_pt, has_lm in train_loader:
            fixed_in = fixed_in.to(device, non_blocking=True)     # (B,1,D,H,W) in [0,1]
            moving_in = moving_in.to(device, non_blocking=True)
            fixed_z = fixed_z.to(device, non_blocking=True)       # (B,1,D,H,W) z-score
            moving_z = moving_z.to(device, non_blocking=True)
            fixed_pt = fixed_pt.to(device, non_blocking=True)     # (B,N,3)
            moving_pt = moving_pt.to(device, non_blocking=True)   # (B,N,3)
            has_lm = has_lm.to(device, non_blocking=True)         # (B,)
            tr_lm_cov += int((has_lm.view(-1) > 0.5).sum().item())
            tr_total += int(has_lm.numel())

            optimizer.zero_grad(set_to_none=True)

            # forward can use AMP
            with torch.cuda.amp.autocast(enabled=use_amp):
                flow_preds = model(fixed_in, moving_in)

            loss = 0.0
            T = len(flow_preds)

            sim_weight = float(wc.sim_weight)
            reg_weight = float(wc.reg_weight)
            jac_weight = float(wc.jac_weight)
            mag_weight = float(wc.mag_weight)
            gamma = float(wc.gamma)
            flow_clamp = float(wc.flow_clamp)
            lm_weight = float(wc.lm_weight)
            lm_iters = int(wc.lm_iters)

            flow_last = None

            for i, flow in enumerate(flow_preds):
                w = gamma ** (T - i - 1)

                # guard against non-finite flow
                if not torch.isfinite(flow).all():
                    raise RuntimeError("Non-finite flow detected (NaN/Inf) before clamp.")

                flow = flow.float().clamp(-flow_clamp, flow_clamp)

                warped_z = warp_3d(moving_z, flow)

                # similarity (FP32), robust fallback
                sim_raw = compute_similarity(
                    fixed_z=fixed_z,
                    warped_z=warped_z,
                    sim_type=str(wc.sim_type),
                    global_ncc=global_ncc,
                    local_ncc=local_ncc,
                    mixed_alpha=float(wc.mixed_alpha),
                )
                if not torch.isfinite(sim_raw):
                    sim_raw = F.mse_loss(fixed_z.float(), warped_z.float())

                # regularization
                smooth = flow_smoothness_l1(flow)
                bend = bending_energy(flow)

                reg = (smooth_frac * smooth) + (bend_frac * bend)

                # optional penalties
                jac = jacobian_det_penalty(flow) if jac_weight > 0.0 else flow.new_tensor(0.0)
                mag = flow_magnitude_l2(flow) if mag_weight > 0.0 else flow.new_tensor(0.0)

                step_loss = (sim_weight * sim_raw) + (reg_weight * reg) + (jac_weight * jac) + (mag_weight * mag)
                loss = loss + w * step_loss

                # accumulate components (weighted)
                tr_sim_sum += float((w * sim_raw).item())
                tr_smooth_sum += float((w * smooth).item())
                tr_bend_sum += float((w * bend).item())
                tr_jac_sum += float((w * jac).item())
                tr_mag_sum += float((w * mag).item())

                if i == (T - 1):
                    flow_last = flow

            # Landmark loss (only on last iteration flow for speed)
            lm = flow_preds[-1].new_tensor(0.0)
            if lm_weight > 0.0 and flow_last is not None:
                valid = (has_lm.view(-1) > 0.5)
                if valid.any():
                    fx_pred = moving_points_to_fixed(flow_last[valid], moving_pt[valid], iters=lm_iters)
                    lm = F.smooth_l1_loss(fx_pred, fixed_pt[valid])
            loss = loss + (lm_weight * lm)
            tr_lm_sum += float(lm.item())
            if lm_weight > 0.0 and (has_lm.view(-1) > 0.5).any():
                valid = (has_lm.view(-1) > 0.5)
                tr_lm_valid_sum += float(lm.item()) * int(valid.sum().item())

            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite total loss detected (NaN/Inf).")

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(wc.grad_clip))
            scaler.step(optimizer)
            scaler.update()

            train_sum += float(loss.item())
            train_batches += 1

        train_loss = train_sum / max(1, train_batches)
        tr_sim = tr_sim_sum / max(1, train_batches)
        tr_smooth = tr_smooth_sum / max(1, train_batches)
        tr_bend = tr_bend_sum / max(1, train_batches)
        tr_jac = tr_jac_sum / max(1, train_batches)
        tr_mag = tr_mag_sum / max(1, train_batches)
        tr_lm = tr_lm_sum / max(1, train_batches)
        tr_lm_coverage = tr_lm_cov / max(1, tr_total)
        tr_lm_valid_only = tr_lm_valid_sum / max(1, tr_lm_cov)

        # ----------------- VAL ----------------- #
        model.eval()
        val_sum = 0.0
        val_batches = 0

        va_sim_sum = 0.0
        va_smooth_sum = 0.0
        va_bend_sum = 0.0
        va_jac_sum = 0.0
        va_mag_sum = 0.0
        va_lm_sum = 0.0
        va_lm_valid_sum = 0.0  # sum(lm) over valid samples only
        va_lm_cov = 0
        va_total = 0

        with torch.no_grad():
            for fixed_in, moving_in, fixed_z, moving_z, fixed_pt, moving_pt, has_lm in val_loader:
                fixed_in = fixed_in.to(device, non_blocking=True)
                moving_in = moving_in.to(device, non_blocking=True)
                fixed_z = fixed_z.to(device, non_blocking=True)
                moving_z = moving_z.to(device, non_blocking=True)
                fixed_pt = fixed_pt.to(device, non_blocking=True)
                moving_pt = moving_pt.to(device, non_blocking=True)
                has_lm = has_lm.to(device, non_blocking=True)
                va_lm_cov += int((has_lm.view(-1) > 0.5).sum().item())
                va_total += int(has_lm.numel())

                with torch.cuda.amp.autocast(enabled=use_amp):
                    flow_preds = model(fixed_in, moving_in)

                flow = flow_preds[-1]
                if not torch.isfinite(flow).all():
                    raise RuntimeError("Non-finite flow detected (NaN/Inf) during validation.")

                flow = flow.float().clamp(-float(wc.flow_clamp), float(wc.flow_clamp))
                warped_z = warp_3d(moving_z, flow)

                sim_raw = compute_similarity(
                    fixed_z=fixed_z,
                    warped_z=warped_z,
                    sim_type=str(wc.sim_type),
                    global_ncc=global_ncc,
                    local_ncc=local_ncc if local_ncc is not None else None,
                    mixed_alpha=float(wc.mixed_alpha),
                )
                if not torch.isfinite(sim_raw):
                    sim_raw = F.mse_loss(fixed_z.float(), warped_z.float())

                smooth = flow_smoothness_l1(flow)
                bend = bending_energy(flow)
                reg = (smooth_frac * smooth) + (bend_frac * bend)

                jac = jacobian_det_penalty(flow) if float(wc.jac_weight) > 0.0 else flow.new_tensor(0.0)
                mag = flow_magnitude_l2(flow) if float(wc.mag_weight) > 0.0 else flow.new_tensor(0.0)

                # Landmark loss (validation, last flow only)
                lm = flow.new_tensor(0.0)
                lm_weight = float(wc.lm_weight)
                if lm_weight > 0.0:
                    valid = (has_lm.view(-1) > 0.5)
                    if valid.any():
                        fx_pred = moving_points_to_fixed(flow[valid], moving_pt[valid], iters=int(wc.lm_iters))
                        lm = F.smooth_l1_loss(fx_pred, fixed_pt[valid])

                val_loss = (
                    float(wc.sim_weight) * sim_raw
                    + float(wc.reg_weight) * reg
                    + float(wc.jac_weight) * jac
                    + float(wc.mag_weight) * mag
                    + float(wc.lm_weight) * lm
                )

                if not torch.isfinite(val_loss):
                    raise RuntimeError("Non-finite validation loss detected (NaN/Inf).")

                val_sum += float(val_loss.item())
                val_batches += 1

                va_sim_sum += float(sim_raw.item())
                va_smooth_sum += float(smooth.item())
                va_bend_sum += float(bend.item())
                va_jac_sum += float(jac.item())
                va_mag_sum += float(mag.item())
                va_lm_sum += float(lm.item())
                if float(wc.lm_weight) > 0.0 and (has_lm.view(-1) > 0.5).any():
                    valid = (has_lm.view(-1) > 0.5)
                    va_lm_valid_sum += float(lm.item()) * int(valid.sum().item())

        val_loss = val_sum / max(1, val_batches)
        va_sim = va_sim_sum / max(1, val_batches)
        va_smooth = va_smooth_sum / max(1, val_batches)
        va_bend = va_bend_sum / max(1, val_batches)
        va_jac = va_jac_sum / max(1, val_batches)
        va_mag = va_mag_sum / max(1, val_batches)
        va_lm = va_lm_sum / max(1, val_batches)
        va_lm_coverage = va_lm_cov / max(1, va_total)
        va_lm_valid_only = va_lm_valid_sum / max(1, va_lm_cov)

        dt_min = (time.time() - t0) / 60.0
        print(f"Epoch {epoch:03d}/{int(wc.epochs):03d}  train={train_loss:.4f}  val={val_loss:.4f}  time={dt_min:.1f}min")

        wandb.log(
            {
                "loss/train": train_loss,
                "loss/val": val_loss,

                "comp/train_sim_wsum": tr_sim,
                "comp/train_smooth_wsum": tr_smooth,
                "comp/train_bend_wsum": tr_bend,
                "comp/train_jac_wsum": tr_jac,
                "comp/train_mag_wsum": tr_mag,
                "comp/train_lm": tr_lm,
                "lm/train_coverage": tr_lm_coverage,
                "lm/train_lm_valid_only": tr_lm_valid_only,

                "comp/val_sim": va_sim,
                "comp/val_smooth": va_smooth,
                "comp/val_bend": va_bend,
                "comp/val_jac": va_jac,
                "comp/val_mag": va_mag,
                "comp/val_lm": va_lm,
                "lm/val_coverage": va_lm_coverage,
                "lm/val_lm_valid_only": va_lm_valid_only,

                "lr": float(wc.lr),
                "reg_weight": float(wc.reg_weight),
                "sim_weight": float(wc.sim_weight),
                "jac_weight": float(wc.jac_weight),
                "mag_weight": float(wc.mag_weight),
                "lm_weight": float(wc.lm_weight),
                "lm_iters": int(wc.lm_iters),
                "gamma": float(wc.gamma),
                "zclamp": float(wc.zclamp),
                "flow_clamp": float(wc.flow_clamp),
                "sim_type": str(wc.sim_type),
                "ncc_win": int(wc.ncc_win),
                "mixed_alpha": float(wc.mixed_alpha),
                "time/epoch_min": dt_min,
            },
            step=epoch,
        )

        # checkpoints
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), ckpt_dir / "best_model.pth")
            wandb.run.summary["best_val_loss"] = best_val
            wandb.run.summary["best_model_path"] = str(ckpt_dir / "best_model.pth")

        if (epoch % int(wc.save_every) == 0) or (epoch == int(wc.epochs)):
            torch.save(model.state_dict(), ckpt_dir / f"epoch_{epoch:03d}.pth")

    print("Training finished. Best val:", best_val)
    print("Exp dir:", exp_dir)
    wandb.finish()


if __name__ == "__main__":
    main()
