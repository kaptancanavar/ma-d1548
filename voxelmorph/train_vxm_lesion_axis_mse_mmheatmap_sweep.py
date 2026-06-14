#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VoxelMorph 3D Training: LESION-ONLY Landmark Supervision (mm) + NIfTI↔JSON Mapping via CSV

This version is SWEEP-ROBUST:
✅ Accepts BOTH hyphen args (CLI) and underscore args (wandb agent auto args)
✅ Accepts list-like strings for triplets:
   --input_shape=[224,224,96]  OR  --input-shape 224 224 96
   --lesion_w_mm=[1,1,3]       OR  --lesion-w-mm 1 1 3
✅ Accepts boolean args as either flags OR explicit values:
   --skip_missing_lesion=True  OR  --skip-missing-lesion
   --log_z_slice_error=True    OR  --log-z-slice-error

Other features (as in your version):
- Axis-wise additive lesion loss (true per-axis contributions): weighted MSE in mm
- Optional mm-metric Gaussian heatmap for anisotropic spacing (--lesion-sigma-mm)
- Keeps lesion_err_mm as unweighted Euclidean mm distance
- Logs axis-wise MSE parts and RMSE per axis
- Optional z error in slices (mean |dz|/sz)
- Only ONE wandb.init()
"""

__author__ = "Semih Tarik Uenal"

import os
import sys
import csv
import time
import json
import argparse
import ast
import re
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Any, Union

import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import wandb

# Force VoxelMorph PyTorch backend
os.environ["NEURITE_BACKEND"] = "pytorch"
os.environ["VXM_BACKEND"] = "pytorch"
import voxelmorph as vxm  # type: ignore


# ---------------- Argparse helpers (SWEEP ROBUST) ---------------- #

def _str2bool(v: Union[str, bool, int, None]) -> bool:
    """Parse booleans from wandb agent style: True/False/1/0/yes/no."""
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    if isinstance(v, int):
        return bool(v)
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def _parse_triplet(v: Any, cast=float, name: str = "triplet") -> List[Any]:
    """
    Accepts:
      - list/tuple length 3
      - argparse list of strings length 1 or 3 (from nargs='+')
      - string like "[1,2,3]" or "(1,2,3)" or "1 2 3" or "1,2,3"
    Returns: list[cast] length 3
    """
    # argparse nargs='+' gives list[str]
    if isinstance(v, list):
        if len(v) == 3:
            return [cast(x) for x in v]
        if len(v) == 1:
            v = v[0]
        else:
            raise ValueError(f"{name} must have 3 values, got {v}")

    if isinstance(v, (tuple, np.ndarray)):
        v = list(v)
        if len(v) != 3:
            raise ValueError(f"{name} must have 3 values, got {v}")
        return [cast(x) for x in v]

    if not isinstance(v, str):
        raise ValueError(f"{name} must be str/list/tuple, got {type(v)}")

    s = v.strip()

    # Try python literal eval first (e.g. "[224, 224, 96]")
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, (list, tuple)) and len(obj) == 3:
            return [cast(x) for x in obj]
    except Exception:
        pass

    # Fallback: strip brackets and split by comma/space
    s2 = s.strip("[]()")
    parts = [p for p in re.split(r"[,\s]+", s2) if p]
    if len(parts) != 3:
        raise ValueError(f"{name} must have 3 values, got '{v}' -> {parts}")
    return [cast(p) for p in parts]


# ---------------- Mapping utils ---------------- #

def relpath_after_study(nii_path: str, study_id: str) -> Path:
    """
    .../<patient>/<study>/<series>/<file>.nii -> <series>/<file>.nii
    """
    p = Path(nii_path)
    parts = p.parts
    try:
        si = parts.index(str(study_id))
    except ValueError:
        raise RuntimeError(f"study_id '{study_id}' not found in path: {nii_path}")
    if si + 1 >= len(parts):
        raise RuntimeError(f"Cannot derive relpath after study_id from: {nii_path}")
    return Path(*parts[si + 1:])


def study_root_from_path(nii_path: str, study_id: str) -> Path:
    """
    .../<patient>/<study>/<...> -> .../<patient>/<study>
    """
    p = Path(nii_path)
    parts = p.parts
    try:
        si = parts.index(str(study_id))
    except ValueError:
        raise RuntimeError(f"study_id '{study_id}' not found in path: {nii_path}")
    return Path(*parts[: si + 1])


def _normalize_rel_nii(rel_nii: str) -> str:
    rel_nii = rel_nii.strip().lstrip("/")
    return rel_nii


class NiiJsonMapper:
    """
    Loads nii_json_mapping.csv and maps (patient_id, study_id, rel_nii) -> annotation_rel
    Handles minor extension mismatches (.nii vs .nii.gz) by trying alternatives.
    """

    def __init__(self, mapping_csv: Path):
        self.mapping_csv = Path(mapping_csv)
        if not self.mapping_csv.is_file():
            raise FileNotFoundError(f"Mapping CSV not found: {self.mapping_csv}")

        self._map: Dict[Tuple[str, str, str], str] = {}
        with self.mapping_csv.open("r", newline="") as f:
            reader = csv.DictReader(f)
            required = {"patient_id", "study_id", "annotation", ".nii"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise RuntimeError(f"Mapping CSV missing columns: {missing}")

            for row in reader:
                pid = str(row["patient_id"]).strip()
                sid = str(row["study_id"]).strip()
                rel_nii = _normalize_rel_nii(str(row[".nii"]))
                ann = str(row["annotation"]).strip().lstrip("/")
                key = (pid, sid, rel_nii)
                if key not in self._map:
                    self._map[key] = ann

    @staticmethod
    def _alt_ext_candidates(rel_nii: str) -> List[str]:
        out = [rel_nii]
        if rel_nii.endswith(".nii.gz"):
            out.append(rel_nii[:-3])  # -> .nii
        elif rel_nii.endswith(".nii"):
            out.append(rel_nii + ".gz")  # -> .nii.gz
        return out

    def get_annotation_rel(self, patient_id: str, study_id: str, rel_nii: Path) -> str:
        pid = str(patient_id).strip()
        sid = str(study_id).strip()
        rel = _normalize_rel_nii(rel_nii.as_posix())

        for cand in self._alt_ext_candidates(rel):
            key = (pid, sid, cand)
            if key in self._map:
                return self._map[key]

        candidates = [k[2] for k in self._map.keys() if k[0] == pid and k[1] == sid]
        msg = (
            f"No mapping entry found for:\n"
            f"  patient_id={pid}\n"
            f"  study_id={sid}\n"
            f"  rel_nii={rel}\n"
        )
        if candidates:
            msg += "  Candidates for this patient+study (first 15):\n    " + "\n    ".join(candidates[:15])
        raise KeyError(msg)


# ---------------- JSON / landmark utils (LESION ONLY) ---------------- #

def _is_lesion_token(s: str) -> bool:
    s = (s or "").lower()
    # NOTE: keep this conservative; avoid overly broad tokens
    return any(k in s for k in ["lesion", "läsion", "laesion", "tumor", "tumour"])


def extract_lesion_mm_from_json(json_path: Path, vol_affine: np.ndarray) -> np.ndarray:
    """
    Returns lesion in world-mm (3,).
    Preference:
      1) coords_transformed
      2) coords
      3) coords_vox -> world via affine
    """
    raw = json.loads(json_path.read_text())
    anns = raw.get("annotations", [])
    lesion_mm: Optional[np.ndarray] = None

    for ann in anns:
        if not isinstance(ann, dict):
            continue
        ann_name = str(ann.get("name", ""))
        pts = ann.get("points", [])
        if not isinstance(pts, list):
            continue

        for pt in pts:
            if not isinstance(pt, dict):
                continue

            pt_name = str(pt.get("name", ""))
            if not (_is_lesion_token(pt_name) or (_is_lesion_token(ann_name) and pt_name.strip() == "")):
                continue

            c_t = pt.get("coords_transformed")
            c = pt.get("coords")

            if isinstance(c_t, (list, tuple)) and len(c_t) >= 3:
                lesion_mm = np.array([float(c_t[0]), float(c_t[1]), float(c_t[2])], dtype=np.float32)
                break
            if isinstance(c, (list, tuple)) and len(c) >= 3:
                lesion_mm = np.array([float(c[0]), float(c[1]), float(c[2])], dtype=np.float32)
                break

            cv = pt.get("coords_vox")
            if isinstance(cv, (list, tuple)) and len(cv) >= 3:
                ijk1 = np.array([float(cv[0]), float(cv[1]), float(cv[2]), 1.0], dtype=np.float32)
                xyz1 = vol_affine @ ijk1
                lesion_mm = xyz1[:3].astype(np.float32)
                break

        if lesion_mm is not None:
            break

    if lesion_mm is None:
        raise RuntimeError(f"Could not find lesion in JSON: {json_path}")

    return lesion_mm


def has_lesion_in_json(json_path: Path) -> bool:
    try:
        raw = json.loads(json_path.read_text())
    except Exception:
        return False

    for ann in raw.get("annotations", []):
        if not isinstance(ann, dict):
            continue
        ann_name = str(ann.get("name", ""))
        pts = ann.get("points", [])
        if not isinstance(pts, list):
            continue
        for pt in pts:
            if not isinstance(pt, dict):
                continue
            pt_name = str(pt.get("name", ""))
            if _is_lesion_token(pt_name) or (_is_lesion_token(ann_name) and pt_name.strip() == ""):
                return True
    return False


def world_to_voxel(affine: np.ndarray, xyz_mm: np.ndarray) -> np.ndarray:
    Ainv = np.linalg.inv(affine)
    xyz1 = np.array([xyz_mm[0], xyz_mm[1], xyz_mm[2], 1.0], dtype=np.float32)
    ijk1 = Ainv @ xyz1
    return ijk1[:3].astype(np.float32)


# ---------------- Heatmap + softargmax (single point) ---------------- #

_GRID_CACHE: Dict[Tuple[Tuple[int, int, int], str, int], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def _grid_key(shape: Tuple[int, int, int], device: torch.device) -> Tuple[Tuple[int, int, int], str, int]:
    dev_idx = int(device.index) if device.type == "cuda" and device.index is not None else -1
    return (tuple(shape), device.type, dev_idx)


def get_xyz_grids(vol_shape: Tuple[int, int, int], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    key = _grid_key(vol_shape, device)
    if key not in _GRID_CACHE:
        X, Y, Z = vol_shape
        xs = torch.arange(X, device=device, dtype=torch.float32).view(1, 1, X, 1, 1)
        ys = torch.arange(Y, device=device, dtype=torch.float32).view(1, 1, 1, Y, 1)
        zs = torch.arange(Z, device=device, dtype=torch.float32).view(1, 1, 1, 1, Z)
        _GRID_CACHE[key] = (xs, ys, zs)
    return _GRID_CACHE[key]


def make_gaussian_heatmap_single_vox(
    pts_vox: torch.Tensor,
    vol_shape: Tuple[int, int, int],
    sigma_vox: float,
) -> torch.Tensor:
    """
    pts_vox: (B,3) voxel coords (x,y,z)
    returns: (B,1,X,Y,Z)
    """
    B = pts_vox.shape[0]
    device = pts_vox.device
    sigma = max(float(sigma_vox), 1e-3)

    xs, ys, zs = get_xyz_grids(vol_shape, device)
    cx = pts_vox[:, 0].view(B, 1, 1, 1, 1)
    cy = pts_vox[:, 1].view(B, 1, 1, 1, 1)
    cz = pts_vox[:, 2].view(B, 1, 1, 1, 1)

    d2 = (xs - cx) ** 2 + (ys - cy) ** 2 + (zs - cz) ** 2
    hm = torch.exp(-0.5 * d2 / (sigma ** 2))
    return hm


def make_gaussian_heatmap_single_mm(
    pts_vox: torch.Tensor,
    vol_shape: Tuple[int, int, int],
    spacing_mm: torch.Tensor,   # (B,3) or (3,)
    sigma_mm: float,
) -> torch.Tensor:
    """
    Build Gaussian using mm-metric distances (handles anisotropic spacing).
    pts_vox: (B,3) voxel coords
    spacing_mm: (B,3) or (3,) with [sx,sy,sz]
    returns: (B,1,X,Y,Z)
    """
    B = pts_vox.shape[0]
    device = pts_vox.device

    sigma2 = max(float(sigma_mm), 1e-6) ** 2

    xs, ys, zs = get_xyz_grids(vol_shape, device)
    cx = pts_vox[:, 0].view(B, 1, 1, 1, 1)
    cy = pts_vox[:, 1].view(B, 1, 1, 1, 1)
    cz = pts_vox[:, 2].view(B, 1, 1, 1, 1)

    if spacing_mm.ndim == 1:
        spacing_mm = spacing_mm.view(1, 3).expand(B, 3)

    sx = spacing_mm[:, 0].view(B, 1, 1, 1, 1)
    sy = spacing_mm[:, 1].view(B, 1, 1, 1, 1)
    sz = spacing_mm[:, 2].view(B, 1, 1, 1, 1)

    dx2 = ((xs - cx) * sx) ** 2
    dy2 = ((ys - cy) * sy) ** 2
    dz2 = ((zs - cz) * sz) ** 2
    d2_mm = dx2 + dy2 + dz2

    hm = torch.exp(-0.5 * d2_mm / sigma2)
    return hm


def softargmax_3d_single(
    heatmap: torch.Tensor,
    beta: float = 2.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    heatmap: (B,1,X,Y,Z) -> returns (B,3) voxel coords (x,y,z)
    """
    B, C, X, Y, Z = heatmap.shape
    if C != 1:
        raise ValueError(f"Expected channel=1 heatmap, got C={C}")

    hm = torch.clamp(heatmap, min=0.0) + eps
    if beta != 1.0:
        hm = hm ** float(beta)

    denom = hm.sum(dim=(2, 3, 4), keepdim=True) + eps
    p = hm / denom  # (B,1,X,Y,Z)

    xs, ys, zs = get_xyz_grids((X, Y, Z), heatmap.device)

    ex = (p * xs).sum(dim=(2, 3, 4))[:, 0]
    ey = (p * ys).sum(dim=(2, 3, 4))[:, 0]
    ez = (p * zs).sum(dim=(2, 3, 4))[:, 0]
    return torch.stack([ex, ey, ez], dim=-1)


def voxel_to_world_torch(aff: torch.Tensor, ijk: torch.Tensor) -> torch.Tensor:
    """
    aff: (B,4,4) or (4,4)
    ijk: (B,3) or (B,1,3)
    returns: (B,3)
    """
    if aff.ndim == 2:
        aff = aff.unsqueeze(0)

    if ijk.ndim == 3 and ijk.shape[1] == 1:
        ijk = ijk[:, 0, :]

    B = ijk.shape[0]
    if aff.shape[0] == 1 and B > 1:
        aff = aff.expand(B, -1, -1)

    ones = torch.ones((B, 1), device=ijk.device, dtype=ijk.dtype)
    ijk1 = torch.cat([ijk, ones], dim=-1)  # (B,4)

    xyz1 = torch.matmul(ijk1, aff.transpose(1, 2))  # (B,4)
    return xyz1[..., :3]


# ---------------- Similarity loss ---------------- #

class GlobalNCCLoss(nn.Module):
    """Global NCC loss = -NCC"""
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.ndim == 5 and x.shape[1] == 1:
            x = x[:, 0, ...]
            y = y[:, 0, ...]
        x = x.reshape(x.shape[0], -1).float()
        y = y.reshape(y.shape[0], -1).float()

        x = x - x.mean(dim=1, keepdim=True)
        y = y - y.mean(dim=1, keepdim=True)

        var_x = (x ** 2).mean(dim=1, keepdim=True)
        var_y = (y ** 2).mean(dim=1, keepdim=True)

        denom = torch.sqrt(var_x * var_y)
        denom = torch.clamp(denom, min=self.eps)

        ncc = (x * y).mean(dim=1, keepdim=True) / denom
        return -ncc.mean()


# ---------------- Dataset ---------------- #

class BreastPairDatasetLesionOnly(Dataset):
    """
    Returns:
      mv: (1,X,Y,Z)
      fx: (1,X,Y,Z)
      mv_les_vox: (3,) moving voxel
      fx_les_mm:  (3,) fixed world-mm
      fx_aff: (4,4) fixed affine
    """

    def __init__(
        self,
        csv_path: Path,
        target_shape: Tuple[int, int, int],
        mapper: NiiJsonMapper,
        skip_missing_lesion: bool = False,
    ):
        super().__init__()
        self.csv_path = Path(csv_path)
        self.target_shape = tuple(target_shape)
        self.mapper = mapper
        self.skip_missing_lesion = bool(skip_missing_lesion)
        self.rows: List[dict] = []

        if not self.csv_path.is_file():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

        kept = 0
        skipped = 0
        skipped_map = 0
        skipped_json = 0
        skipped_no_les = 0

        with self.csv_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                mv = row.get("moving")
                fx = row.get("fixed")
                pid = row.get("patient_id")
                sid = row.get("study_id")
                if not mv or not fx or not pid or not sid:
                    continue

                if self.skip_missing_lesion:
                    try:
                        mv_json = self._json_for_nii(mv, str(pid), str(sid))
                        fx_json = self._json_for_nii(fx, str(pid), str(sid))
                    except Exception:
                        skipped += 1
                        skipped_map += 1
                        continue

                    if (not mv_json.exists()) or (not fx_json.exists()):
                        skipped += 1
                        skipped_json += 1
                        continue

                    has_mv = has_lesion_in_json(mv_json)
                    has_fx = has_lesion_in_json(fx_json)
                    if not (has_mv and has_fx):
                        skipped += 1
                        skipped_no_les += 1
                        continue

                self.rows.append(row)
                kept += 1

        if not self.rows:
            raise RuntimeError(f"No valid rows found in {self.csv_path} after filtering.")

        print(f"[Dataset] {self.csv_path} -> {len(self.rows)} pairs")
        if self.skip_missing_lesion:
            print(
                f"          filtered out: {skipped} (mapping {skipped_map}, missing_json {skipped_json}, no_lesion {skipped_no_les})"
            )

    def __len__(self) -> int:
        return len(self.rows)

    @staticmethod
    def _load_nifti(path: str) -> Tuple[np.ndarray, np.ndarray]:
        img = nib.load(path)
        data = img.get_fdata().astype(np.float32)
        affine = img.affine.astype(np.float32)

        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        data = np.clip(data, -10.0, 10.0)

        if data.ndim != 3:
            raise ValueError(f"Expected 3D volume, got shape {data.shape} at {path}")
        return data, affine

    def _json_for_nii(self, nii_path: str, patient_id: str, study_id: str) -> Path:
        rel_nii = relpath_after_study(nii_path, study_id)
        ann_rel = self.mapper.get_annotation_rel(patient_id, study_id, rel_nii)
        study_root = study_root_from_path(nii_path, study_id)
        return (study_root / ann_rel).resolve()

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        mv_path = row["moving"]
        fx_path = row["fixed"]
        patient_id = str(row["patient_id"])
        study_id = str(row["study_id"])

        mv_vol, mv_aff = self._load_nifti(mv_path)
        fx_vol, fx_aff = self._load_nifti(fx_path)

        if mv_vol.shape != self.target_shape:
            raise ValueError(f"Moving volume shape {mv_vol.shape} != target {self.target_shape} at {mv_path}")
        if fx_vol.shape != self.target_shape:
            raise ValueError(f"Fixed volume shape {fx_vol.shape} != target {self.target_shape} at {fx_path}")

        mv_json = self._json_for_nii(mv_path, patient_id, study_id)
        fx_json = self._json_for_nii(fx_path, patient_id, study_id)

        if not mv_json.exists():
            raise FileNotFoundError(f"Missing moving JSON via mapping: {mv_json}")
        if not fx_json.exists():
            raise FileNotFoundError(f"Missing fixed JSON via mapping: {fx_json}")

        mv_les_mm = extract_lesion_mm_from_json(mv_json, mv_aff)
        fx_les_mm = extract_lesion_mm_from_json(fx_json, fx_aff)

        mv_les_vox = world_to_voxel(mv_aff, mv_les_mm)

        X, Y, Z = self.target_shape
        mv_les_vox = np.clip(mv_les_vox, [0, 0, 0], [X - 1, Y - 1, Z - 1]).astype(np.float32)

        mv = torch.from_numpy(mv_vol).unsqueeze(0)
        fx = torch.from_numpy(fx_vol).unsqueeze(0)

        return (
            mv,
            fx,
            torch.from_numpy(mv_les_vox),                    # (3,)
            torch.from_numpy(fx_les_mm.astype(np.float32)),  # (3,)
            torch.from_numpy(fx_aff.astype(np.float32)),     # (4,4)
        )


# ---------------- Args / misc ---------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train VoxelMorph with LESION-only landmark loss (mm) + mapping CSV.")

    # Required paths (accept hyphen + underscore)
    p.add_argument("--train-csv", "--train_csv", dest="train_csv", type=str, required=True)
    p.add_argument("--val-csv", "--val_csv", dest="val_csv", type=str, required=True)
    p.add_argument("--nii-json-mapping-csv", "--nii_json_mapping_csv", dest="nii_json_mapping_csv", type=str, required=True)
    p.add_argument("--out-root", "--out_root", dest="out_root", type=str, required=True)

    # Core
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=1)
    p.add_argument("--num-workers", "--num_workers", dest="num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)

    p.add_argument("--image-loss", "--image_loss", dest="image_loss", type=str, default="ncc", choices=["ncc", "mse"])
    p.add_argument("--sim-weight", "--sim_weight", dest="sim_weight", type=float, default=1.0)
    p.add_argument("--reg-weight", "--reg_weight", dest="reg_weight", type=float, default=1.0)
    p.add_argument("--grad-downsample", "--grad_downsample", dest="grad_downsample", type=int, default=1)

    # Triplet: accept both "224 224 96" and "[224,224,96]"
    p.add_argument("--input-shape", "--input_shape", dest="input_shape",
                   nargs="+", default=["224", "224", "96"])

    p.add_argument("--save-every", "--save_every", dest="save_every", type=int, default=10)

    p.add_argument("--bidir", type=_str2bool, nargs="?", const=True, default=False)
    p.add_argument("--int-steps", "--int_steps", dest="int_steps", type=int, default=0)
    p.add_argument("--exp-id", "--exp_id", dest="exp_id", type=str, default="")

    # LESION-only supervision
    p.add_argument("--lesion-weight", "--lesion_weight", dest="lesion_weight", type=float, default=0.2)

    # Heatmap parameters (choose ONE mode)
    p.add_argument("--lesion-sigma-vox", "--lesion_sigma_vox", dest="lesion_sigma_vox", type=float, default=2.0,
                   help="Voxel-sigma for isotropic voxel Gaussian heatmap (legacy).")
    p.add_argument("--lesion-sigma-mm", "--lesion_sigma_mm", dest="lesion_sigma_mm", type=float, default=None,
                   help="If set: build lesion heatmap using mm-metric Gaussian with this sigma in mm "
                        "(recommended for anisotropic spacing). Overrides lesion-sigma-vox.")

    p.add_argument("--lesion-beta", "--lesion_beta", dest="lesion_beta", type=float, default=2.0)

    # Triplet: accept both "1 1 3" and "[1,1,3]"
    p.add_argument("--lesion-w-mm", "--lesion_w_mm", dest="lesion_w_mm",
                   nargs="+", default=["1.0", "1.0", "1.0"],
                   help="Axis weights for lesion loss in mm: wx wy wz. Metric lesion_err_mm stays unweighted.")

    p.add_argument("--lesion-loss-mode", "--lesion_loss_mode", dest="lesion_loss_mode",
                   type=str, default="axis_mse", choices=["axis_mse"],
                   help="Lesion supervision loss. 'axis_mse' = sum_i mean(wi * di^2) in mm.")
    p.add_argument("--charbonnier-eps", "--charbonnier_eps", dest="charbonnier_eps", type=float, default=1e-6)

    # Optional 2-phase schedule
    p.add_argument("--phase2-epoch", "--phase2_epoch", dest="phase2_epoch", type=int, default=0,
                   help="If >0: from this epoch on use *_2 values (weights/sigma/beta).")
    p.add_argument("--sim-weight2", "--sim_weight2", dest="sim_weight2", type=float, default=None)
    p.add_argument("--reg-weight2", "--reg_weight2", dest="reg_weight2", type=float, default=None)
    p.add_argument("--lesion-weight2", "--lesion_weight2", dest="lesion_weight2", type=float, default=None)
    p.add_argument("--lesion-sigma-vox2", "--lesion_sigma_vox2", dest="lesion_sigma_vox2", type=float, default=None)
    p.add_argument("--lesion-sigma-mm2", "--lesion_sigma_mm2", dest="lesion_sigma_mm2", type=float, default=None)
    p.add_argument("--lesion-beta2", "--lesion_beta2", dest="lesion_beta2", type=float, default=None)

    # Boolean flags need to accept "=True" from wandb agent:
    p.add_argument("--skip-missing-lesion", "--skip_missing_lesion", dest="skip_missing_lesion",
                   type=_str2bool, nargs="?", const=True, default=False,
                   help="Filter out pairs where lesion is missing in mv or fx JSON.")

    p.add_argument("--wandb-project", "--wandb_project", dest="wandb_project", type=str, default="")

    # Early stopping
    p.add_argument("--early-stop-metric", "--early_stop_metric", dest="early_stop_metric",
                   type=str, default="lesion_err", choices=["lesion_err", "val_loss"],
                   help="Metric to monitor for early stopping.")
    p.add_argument("--early-stop-patience", "--early_stop_patience", dest="early_stop_patience", type=int, default=20,
                   help="Stop after N epochs without improvement. Set 0 to disable.")
    p.add_argument("--early-stop-min-delta", "--early_stop_min_delta", dest="early_stop_min_delta", type=float, default=1e-4,
                   help="Required improvement to reset patience.")

    p.add_argument("--log-z-slice-error", "--log_z_slice_error", dest="log_z_slice_error",
                   type=_str2bool, nargs="?", const=True, default=False,
                   help="Log mean |dz| / sz as lesion_err_z_slices.")

    args = p.parse_args()

    # Post-process triplets
    args.input_shape = _parse_triplet(args.input_shape, cast=int, name="input_shape")
    args.lesion_w_mm = _parse_triplet(args.lesion_w_mm, cast=float, name="lesion_w_mm")

    return args


def create_experiment_dir(out_root: Path) -> Path:
    ts = datetime.now().strftime("run_%y%m%d_%H%M%S")
    exp_dir = out_root / ts
    (exp_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (exp_dir / "logs").mkdir(parents=True, exist_ok=True)
    return exp_dir


def save_config(exp_dir: Path, args: argparse.Namespace, device: torch.device):
    cfg_path = exp_dir / "config.txt"
    with cfg_path.open("w") as f:
        f.write(f"Created: {datetime.now().isoformat()}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Python: {sys.version}\n")
        f.write(f"PyTorch: {torch.__version__}\n")
        f.write(f"VoxelMorph: {vxm.__version__ if hasattr(vxm, '__version__') else 'unknown'}\n\n")
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")


def current_phase_params(args: argparse.Namespace, epoch: int) -> Dict[str, float]:
    sim_w = float(args.sim_weight)
    reg_w = float(args.reg_weight)
    les_w = float(args.lesion_weight)
    sigma_vox = float(args.lesion_sigma_vox)
    beta = float(args.lesion_beta)

    sigma_mm = args.lesion_sigma_mm if args.lesion_sigma_mm is not None else None

    if args.phase2_epoch and epoch >= int(args.phase2_epoch):
        if args.sim_weight2 is not None:
            sim_w = float(args.sim_weight2)
        if args.reg_weight2 is not None:
            reg_w = float(args.reg_weight2)
        if args.lesion_weight2 is not None:
            les_w = float(args.lesion_weight2)
        if args.lesion_sigma_vox2 is not None:
            sigma_vox = float(args.lesion_sigma_vox2)
        if args.lesion_sigma_mm2 is not None:
            sigma_mm = float(args.lesion_sigma_mm2)
        if args.lesion_beta2 is not None:
            beta = float(args.lesion_beta2)

    out = {
        "sim_weight": sim_w,
        "reg_weight": reg_w,
        "lesion_weight": les_w,
        "lesion_sigma_vox": sigma_vox,
        "lesion_beta": beta,
    }
    if sigma_mm is not None:
        out["lesion_sigma_mm"] = float(sigma_mm)
    return out


# ---------------- Main training ---------------- #

def main():
    args = parse_args()

    train_csv = Path(args.train_csv).resolve()
    val_csv = Path(args.val_csv).resolve()
    map_csv = Path(args.nii_json_mapping_csv).resolve()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Device: {device}")
    if use_cuda:
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # W&B init (single init!)
    wandb_project = args.wandb_project or ("vxm-ncc-lesion" if args.image_loss == "ncc" else "vxm-lesion-mse")
    run = wandb.init(
        project=wandb_project,
        config={k: v for k, v in vars(args).items()},
    )
    run.name = (f"{args.exp_id}_" if args.exp_id else "") + f"vxm_{args.image_loss}_lesiononly"

    # Sweep override: overwrite args from wandb.config
    cfg = wandb.config
    overridden = {}
    for k in cfg.keys():
        if hasattr(args, k):
            old_v = getattr(args, k)
            new_v = cfg[k]

            # keep lesion_w_mm always list[float]
            if k == "lesion_w_mm" and new_v is not None:
                new_v = _parse_triplet(new_v, cast=float, name="lesion_w_mm")

            # keep input_shape always list[int]
            if k == "input_shape" and new_v is not None:
                new_v = _parse_triplet(new_v, cast=int, name="input_shape")

            # booleans may come as True/False already
            if k in ("skip_missing_lesion", "log_z_slice_error", "bidir"):
                new_v = _str2bool(new_v)

            setattr(args, k, new_v)
            if old_v != new_v:
                overridden[k] = {"old": old_v, "new": new_v}

    if overridden:
        print("[W&B] Sweep overrides applied:")
        for k, v in overridden.items():
            print(f"  - {k}: {v['old']} -> {v['new']}")
        wandb.run.summary["sweep_overrides"] = overridden
    else:
        print("[W&B] No sweep overrides applied (running with CLI args).")

    wandb.config.update(vars(args), allow_val_change=True)

    # Re-resolve after overrides
    train_csv = Path(args.train_csv).resolve()
    val_csv = Path(args.val_csv).resolve()
    map_csv = Path(args.nii_json_mapping_csv).resolve()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # Seeds / cudnn
    torch.manual_seed(42)
    np.random.seed(42)
    if use_cuda:
        torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = True

    input_shape = tuple(args.input_shape)

    mapper = NiiJsonMapper(map_csv)

    train_ds = BreastPairDatasetLesionOnly(
        train_csv, target_shape=input_shape, mapper=mapper, skip_missing_lesion=args.skip_missing_lesion
    )
    val_ds = BreastPairDatasetLesionOnly(
        val_csv, target_shape=input_shape, mapper=mapper, skip_missing_lesion=args.skip_missing_lesion
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=use_cuda
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=use_cuda
    )

    # Model
    enc_nf = [16, 32, 32, 32]
    dec_nf = [32, 32, 32, 32, 32, 16]
    nb_features = [enc_nf, dec_nf]

    model = vxm.networks.VxmDense(
        inshape=input_shape,
        nb_unet_features=nb_features,
        int_steps=args.int_steps,
        int_downsize=2,
        bidir=bool(args.bidir),
    ).to(device)

    transformer = getattr(model, "transformer", None)
    if transformer is None:
        transformer = vxm.layers.SpatialTransformer(inshape=input_shape).to(device)

    # Losses
    mse_loss_fn = vxm.losses.MSE().loss
    base_sim_loss_fn = GlobalNCCLoss() if args.image_loss == "ncc" else mse_loss_fn
    reg_loss_fn = vxm.losses.Grad("l2", loss_mult=args.grad_downsample).loss

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # AMP only for MSE on CUDA
    use_autocast = bool(use_cuda and (args.image_loss == "mse"))
    autocast_device = "cuda" if use_cuda else "cpu"
    scaler: Optional[torch.amp.GradScaler] = torch.amp.GradScaler("cuda") if use_autocast else None

    exp_dir = create_experiment_dir(out_root)
    ckpt_dir = exp_dir / "checkpoints"
    log_dir = exp_dir / "logs"
    save_config(exp_dir, args, device)
    writer = SummaryWriter(log_dir=str(log_dir))
    wandb.run.summary["exp_dir"] = str(exp_dir)

    best_val = float("inf")
    best_val_epoch = 0
    best_lesion_err = float("inf")
    best_lesion_epoch = 0

    es_best = float("inf")
    es_best_epoch = 0
    es_bad_epochs = 0

    print(f"Experiment dir: {exp_dir}")
    print("Start training...\n")

    spacing_logged = False

    def lesion_loss_and_err_axiswise(pred_mm: torch.Tensor, gt_mm: torch.Tensor, eps: float):
        diff = pred_mm - gt_mm                          # (B,3)
        absdiff = diff.abs()
        mae_xyz = absdiff.mean(dim=0)                   # (3,)

        err = torch.linalg.norm(diff, dim=-1)           # (B,)
        err_mean = err.mean()

        wx, wy, wz = [float(v) for v in args.lesion_w_mm]
        w = torch.tensor([wx, wy, wz], device=diff.device, dtype=diff.dtype)

        diff2 = diff * diff                             # (B,3)
        l2_unw_xyz = diff2.mean(dim=0)                  # (3,)
        l2_w_xyz = (diff2 * w).mean(dim=0)              # (3,)
        rmse_unw_xyz = torch.sqrt(l2_unw_xyz + eps)     # (3,)

        loss_raw = l2_w_xyz.sum()
        return loss_raw, err_mean, mae_xyz, l2_unw_xyz, l2_w_xyz, rmse_unw_xyz

    def run_epoch(loader, train: bool, params: Dict[str, float], epoch: int) -> Dict[str, float]:
        nonlocal spacing_logged
        model.train(train)

        sums: Dict[str, float] = {
            "loss": 0.0,
            "sim": 0.0,
            "reg": 0.0,
            "lesion": 0.0,
            "lesion_err_mm": 0.0,
            "lesion_mae_x_mm": 0.0,
            "lesion_mae_y_mm": 0.0,
            "lesion_mae_z_mm": 0.0,
            "lesion_l2_unw_x": 0.0,
            "lesion_l2_unw_y": 0.0,
            "lesion_l2_unw_z": 0.0,
            "lesion_l2_w_x": 0.0,
            "lesion_l2_w_y": 0.0,
            "lesion_l2_w_z": 0.0,
            "lesion_rmse_x_mm": 0.0,
            "lesion_rmse_y_mm": 0.0,
            "lesion_rmse_z_mm": 0.0,
        }

        if args.log_z_slice_error:
            sums["lesion_err_z_slices"] = 0.0

        nb = 0

        ctx: Any = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for mv, fx, mv_les_vox, fx_les_mm, fx_aff in loader:
                mv = mv.to(device)
                fx = fx.to(device)
                mv_les_vox = mv_les_vox.to(device)   # (B,3)
                fx_les_mm = fx_les_mm.to(device)     # (B,3)
                fx_aff = fx_aff.to(device)           # (B,4,4)

                # spacing from affine (per batch)
                A = fx_aff[:, :3, :3]  # (B,3,3)
                sx = torch.linalg.norm(A[:, :, 0], dim=1)  # (B,)
                sy = torch.linalg.norm(A[:, :, 1], dim=1)
                sz = torch.linalg.norm(A[:, :, 2], dim=1)
                spacing_b = torch.stack([sx, sy, sz], dim=1)  # (B,3)

                if (not train) and (not spacing_logged):
                    print(f"[Spacing] fx voxel size approx (mm): sx={float(sx[0]):.4f}, sy={float(sy[0]):.4f}, sz={float(sz[0]):.4f}")
                    wandb.log(
                        {"data/fx_spacing_x_mm": float(sx[0]), "data/fx_spacing_y_mm": float(sy[0]), "data/fx_spacing_z_mm": float(sz[0])},
                        step=epoch,
                    )
                    wandb.run.summary["fx_spacing_x_mm"] = float(sx[0])
                    wandb.run.summary["fx_spacing_y_mm"] = float(sy[0])
                    wandb.run.summary["fx_spacing_z_mm"] = float(sz[0])
                    spacing_logged = True

                if train:
                    optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(autocast_device, enabled=use_autocast):
                    out = model(mv, fx)
                    if bool(args.bidir):
                        warp_m2f, warp_f2m, flow_m2f, flow_f2m = out
                        warp_used = warp_m2f
                        flow_used = flow_m2f
                    else:
                        warp_used, flow_used = out

                    sim_raw = base_sim_loss_fn(fx, warp_used)
                    if not torch.isfinite(sim_raw):
                        sim_raw = mse_loss_fn(fx, warp_used)
                    sim_loss = sim_raw * params["sim_weight"]

                    reg_loss = reg_loss_fn(None, flow_used) * params["reg_weight"]

                    # heatmap build: mm-metric if sigma_mm is set, else voxel-sigma
                    if "lesion_sigma_mm" in params:
                        hm_mv = make_gaussian_heatmap_single_mm(
                            mv_les_vox, input_shape, spacing_b, sigma_mm=float(params["lesion_sigma_mm"])
                        )
                    else:
                        hm_mv = make_gaussian_heatmap_single_vox(
                            mv_les_vox, input_shape, sigma_vox=float(params["lesion_sigma_vox"])
                        )

                    hm_fx = transformer(hm_mv, flow_used)

                    pred_vox = softargmax_3d_single(hm_fx, beta=params["lesion_beta"])
                    pred_mm = voxel_to_world_torch(fx_aff, pred_vox)

                    if pred_mm.ndim == 3 and pred_mm.shape[1] == 1:
                        pred_mm = pred_mm[:, 0, :]

                    lesion_raw, lesion_err_mean, mae_xyz, l2_unw_xyz, l2_w_xyz, rmse_xyz = lesion_loss_and_err_axiswise(
                        pred_mm, fx_les_mm, eps=float(args.charbonnier_eps)
                    )
                    lesion_loss = lesion_raw * params["lesion_weight"]

                    loss = sim_loss + reg_loss + lesion_loss

                if train:
                    if not torch.isfinite(loss):
                        raise RuntimeError("Non-finite loss in TRAIN.")

                    if use_autocast:
                        assert scaler is not None
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()

                sums["loss"] += float(loss.item())
                sums["sim"] += float(sim_loss.item())
                sums["reg"] += float(reg_loss.item())
                sums["lesion"] += float(lesion_loss.item())
                sums["lesion_err_mm"] += float(lesion_err_mean.item())

                sums["lesion_mae_x_mm"] += float(mae_xyz[0].item())
                sums["lesion_mae_y_mm"] += float(mae_xyz[1].item())
                sums["lesion_mae_z_mm"] += float(mae_xyz[2].item())

                sums["lesion_l2_unw_x"] += float(l2_unw_xyz[0].item())
                sums["lesion_l2_unw_y"] += float(l2_unw_xyz[1].item())
                sums["lesion_l2_unw_z"] += float(l2_unw_xyz[2].item())

                sums["lesion_l2_w_x"] += float(l2_w_xyz[0].item())
                sums["lesion_l2_w_y"] += float(l2_w_xyz[1].item())
                sums["lesion_l2_w_z"] += float(l2_w_xyz[2].item())

                sums["lesion_rmse_x_mm"] += float(rmse_xyz[0].item())
                sums["lesion_rmse_y_mm"] += float(rmse_xyz[1].item())
                sums["lesion_rmse_z_mm"] += float(rmse_xyz[2].item())

                if args.log_z_slice_error:
                    dz = (pred_mm[:, 2] - fx_les_mm[:, 2]).abs()
                    sums["lesion_err_z_slices"] += float((dz / (sz + 1e-8)).mean().item())

                nb += 1

        return {k: v / max(nb, 1) for k, v in sums.items()}

    last_val_avg = {"loss": float("inf"), "lesion_err_mm": float("inf")}
    for epoch in range(1, args.epochs + 1):
        params = current_phase_params(args, epoch)

        t0 = time.time()
        train_avg = run_epoch(train_loader, train=True, params=params, epoch=epoch)
        val_avg = run_epoch(val_loader, train=False, params=params, epoch=epoch)
        last_val_avg = val_avg
        elapsed = time.time() - t0

        extra = ""
        if args.log_z_slice_error:
            extra = f" - z_slices={val_avg['lesion_err_z_slices']:.3f}"

        hm_mode = "mm" if ("lesion_sigma_mm" in params) else "vox"
        hm_sigma = params["lesion_sigma_mm"] if ("lesion_sigma_mm" in params) else params["lesion_sigma_vox"]

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} "
            f"- train_loss {train_avg['loss']:.4f} (sim {train_avg['sim']:.4f}, reg {train_avg['reg']:.4f}, lesion {train_avg['lesion']:.4f}) "
            f"- val_loss {val_avg['loss']:.4f} (sim {val_avg['sim']:.4f}, reg {val_avg['reg']:.4f}, lesion {val_avg['lesion']:.4f}) "
            f"- lesion_err {val_avg['lesion_err_mm']:.2f} mm "
            f"- mae(x,y,z)=({val_avg['lesion_mae_x_mm']:.2f},{val_avg['lesion_mae_y_mm']:.2f},{val_avg['lesion_mae_z_mm']:.2f}) mm "
            f"- l2w(x,y,z)=({val_avg['lesion_l2_w_x']:.4f},{val_avg['lesion_l2_w_y']:.4f},{val_avg['lesion_l2_w_z']:.4f}) "
            f"- rmse_z={val_avg['lesion_rmse_z_mm']:.3f} mm"
            f"{extra} "
            f"- hm={hm_mode}(sigma={hm_sigma}) "
            f"- time {elapsed/60:.1f} min"
        )

        # TensorBoard
        writer.add_scalar("Loss/train_total", train_avg["loss"], epoch)
        writer.add_scalar("Loss/val_total", val_avg["loss"], epoch)

        writer.add_scalar("Metric/train_lesion_err_mm", train_avg["lesion_err_mm"], epoch)
        writer.add_scalar("Metric/val_lesion_err_mm", val_avg["lesion_err_mm"], epoch)

        writer.add_scalar("Metric/train_lesion_mae_x_mm", train_avg["lesion_mae_x_mm"], epoch)
        writer.add_scalar("Metric/train_lesion_mae_y_mm", train_avg["lesion_mae_y_mm"], epoch)
        writer.add_scalar("Metric/train_lesion_mae_z_mm", train_avg["lesion_mae_z_mm"], epoch)
        writer.add_scalar("Metric/val_lesion_mae_x_mm", val_avg["lesion_mae_x_mm"], epoch)
        writer.add_scalar("Metric/val_lesion_mae_y_mm", val_avg["lesion_mae_y_mm"], epoch)
        writer.add_scalar("Metric/val_lesion_mae_z_mm", val_avg["lesion_mae_z_mm"], epoch)

        writer.add_scalar("AxisLoss/val_l2_unw_x", val_avg["lesion_l2_unw_x"], epoch)
        writer.add_scalar("AxisLoss/val_l2_unw_y", val_avg["lesion_l2_unw_y"], epoch)
        writer.add_scalar("AxisLoss/val_l2_unw_z", val_avg["lesion_l2_unw_z"], epoch)
        writer.add_scalar("AxisLoss/val_l2w_x", val_avg["lesion_l2_w_x"], epoch)
        writer.add_scalar("AxisLoss/val_l2w_y", val_avg["lesion_l2_w_y"], epoch)
        writer.add_scalar("AxisLoss/val_l2w_z", val_avg["lesion_l2_w_z"], epoch)
        writer.add_scalar("AxisLoss/val_rmse_x_mm", val_avg["lesion_rmse_x_mm"], epoch)
        writer.add_scalar("AxisLoss/val_rmse_y_mm", val_avg["lesion_rmse_y_mm"], epoch)
        writer.add_scalar("AxisLoss/val_rmse_z_mm", val_avg["lesion_rmse_z_mm"], epoch)

        if args.log_z_slice_error:
            writer.add_scalar("Metric/val_lesion_err_z_slices", val_avg["lesion_err_z_slices"], epoch)

        # W&B log
        wandb_payload = {
            "epoch": epoch,
            "phase/sim_weight": params["sim_weight"],
            "phase/reg_weight": params["reg_weight"],
            "phase/lesion_weight": params["lesion_weight"],
            "phase/lesion_beta": params["lesion_beta"],
            "lesion/axis_wx": float(args.lesion_w_mm[0]),
            "lesion/axis_wy": float(args.lesion_w_mm[1]),
            "lesion/axis_wz": float(args.lesion_w_mm[2]),
            "heatmap/mode": hm_mode,
            "heatmap/sigma": float(hm_sigma),
            "loss/train_total": train_avg["loss"],
            "loss/train_sim": train_avg["sim"],
            "loss/train_reg": train_avg["reg"],
            "loss/train_lesion": train_avg["lesion"],
            "loss/val_total": val_avg["loss"],
            "loss/val_sim": val_avg["sim"],
            "loss/val_reg": val_avg["reg"],
            "loss/val_lesion": val_avg["lesion"],
            "metric/train_lesion_err_mm": train_avg["lesion_err_mm"],
            "metric/val_lesion_err_mm": val_avg["lesion_err_mm"],
            "metric/val_lesion_mae_x_mm": val_avg["lesion_mae_x_mm"],
            "metric/val_lesion_mae_y_mm": val_avg["lesion_mae_y_mm"],
            "metric/val_lesion_mae_z_mm": val_avg["lesion_mae_z_mm"],
            "axisloss/val_l2_unw_x": val_avg["lesion_l2_unw_x"],
            "axisloss/val_l2_unw_y": val_avg["lesion_l2_unw_y"],
            "axisloss/val_l2_unw_z": val_avg["lesion_l2_unw_z"],
            "axisloss/val_l2w_x": val_avg["lesion_l2_w_x"],
            "axisloss/val_l2w_y": val_avg["lesion_l2_w_y"],
            "axisloss/val_l2w_z": val_avg["lesion_l2_w_z"],
            "axisloss/val_rmse_x_mm": val_avg["lesion_rmse_x_mm"],
            "axisloss/val_rmse_y_mm": val_avg["lesion_rmse_y_mm"],
            "axisloss/val_rmse_z_mm": val_avg["lesion_rmse_z_mm"],
            "time/epoch_min": elapsed / 60.0,
        }
        if args.log_z_slice_error:
            wandb_payload["metric/val_lesion_err_z_slices"] = val_avg["lesion_err_z_slices"]

        wandb.log(wandb_payload, step=epoch)

        # Save best by val_loss
        if val_avg["loss"] < best_val:
            best_val = float(val_avg["loss"])
            best_val_epoch = epoch
            best_path_loss = ckpt_dir / "best_by_val_loss.pth"
            torch.save(model.state_dict(), best_path_loss)
            print(f"  -> New best (by val_loss) saved to {best_path_loss} (val_loss={best_val:.4f})")

            wandb.run.summary["best_val_loss"] = best_val
            wandb.run.summary["best_val_loss_epoch"] = best_val_epoch
            wandb.run.summary["best_by_val_loss_path"] = str(best_path_loss)

        # Save best by lesion_err_mm
        if val_avg["lesion_err_mm"] < (best_lesion_err - float(args.early_stop_min_delta)):
            best_lesion_err = float(val_avg["lesion_err_mm"])
            best_lesion_epoch = epoch
            best_path_les = ckpt_dir / "best_by_lesion_err_mm.pth"
            torch.save(model.state_dict(), best_path_les)
            print(f"  -> New best (by lesion_err) saved to {best_path_les} (lesion_err={best_lesion_err:.3f} mm)")

            wandb.run.summary["best_lesion_err_mm"] = best_lesion_err
            wandb.run.summary["best_lesion_err_epoch"] = best_lesion_epoch
            wandb.run.summary["best_by_lesion_err_path"] = str(best_path_les)

        # Periodic checkpoint
        if (epoch % args.save_every == 0) or (epoch == args.epochs):
            ckpt_path = ckpt_dir / f"checkpoint_epoch_{epoch:03d}.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss": float(val_avg["loss"]),
                    "val_lesion_err_mm": float(val_avg["lesion_err_mm"]),
                    "val_lesion_mae_x_mm": float(val_avg["lesion_mae_x_mm"]),
                    "val_lesion_mae_y_mm": float(val_avg["lesion_mae_y_mm"]),
                    "val_lesion_mae_z_mm": float(val_avg["lesion_mae_z_mm"]),
                    "val_lesion_l2_unw_x": float(val_avg["lesion_l2_unw_x"]),
                    "val_lesion_l2_unw_y": float(val_avg["lesion_l2_unw_y"]),
                    "val_lesion_l2_unw_z": float(val_avg["lesion_l2_unw_z"]),
                    "val_lesion_l2w_x": float(val_avg["lesion_l2_w_x"]),
                    "val_lesion_l2w_y": float(val_avg["lesion_l2_w_y"]),
                    "val_lesion_l2w_z": float(val_avg["lesion_l2_w_z"]),
                    "val_lesion_rmse_x_mm": float(val_avg["lesion_rmse_x_mm"]),
                    "val_lesion_rmse_y_mm": float(val_avg["lesion_rmse_y_mm"]),
                    "val_lesion_rmse_z_mm": float(val_avg["lesion_rmse_z_mm"]),
                    "phase_params": params,
                    "lesion_w_mm": [float(x) for x in args.lesion_w_mm],
                    "heatmap_mode": "mm" if ("lesion_sigma_mm" in params) else "vox",
                    "heatmap_sigma": float(hm_sigma),
                },
                ckpt_path,
            )
            print(f"  -> Checkpoint saved to {ckpt_path}")

        # Early stopping bookkeeping
        if args.early_stop_metric == "lesion_err":
            cur = float(val_avg["lesion_err_mm"])
            metric_name = "lesion_err_mm"
        else:
            cur = float(val_avg["loss"])
            metric_name = "val_loss"

        if cur < (es_best - float(args.early_stop_min_delta)):
            es_best = cur
            es_best_epoch = epoch
            es_bad_epochs = 0
        else:
            es_bad_epochs += 1

        wandb.log(
            {
                "early_stop/best_metric": es_best,
                "early_stop/best_epoch": es_best_epoch,
                "early_stop/bad_epochs": es_bad_epochs,
            },
            step=epoch,
        )

        if args.early_stop_patience > 0 and es_bad_epochs >= args.early_stop_patience:
            ckpt_path = ckpt_dir / f"checkpoint_early_stop_epoch_{epoch:03d}.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss": float(val_avg["loss"]),
                    "val_lesion_err_mm": float(val_avg["lesion_err_mm"]),
                    "early_stop_metric": args.early_stop_metric,
                    "early_stop_best_metric": es_best,
                    "early_stop_best_epoch": es_best_epoch,
                },
                ckpt_path,
            )
            print(f"  -> Early-stop checkpoint saved to {ckpt_path}")
            print(
                f"Early stopping: no improvement in {metric_name} for {es_bad_epochs} epochs "
                f"(best {metric_name}={es_best:.6f} at epoch {es_best_epoch})."
            )
            break

    writer.close()
    wandb.run.summary["final_epoch"] = epoch
    wandb.run.summary["final_val_loss"] = float(last_val_avg["loss"])
    wandb.run.summary["final_val_lesion_err_mm"] = float(last_val_avg["lesion_err_mm"])
    wandb.run.summary["best_val_loss"] = best_val
    wandb.run.summary["best_val_loss_epoch"] = best_val_epoch
    wandb.run.summary["best_lesion_err_mm"] = best_lesion_err
    wandb.run.summary["best_lesion_err_epoch"] = best_lesion_epoch

    wandb.finish()
    print("\nTraining finished.")
    print(f"Experiment directory: {exp_dir}")
    print(f"Best val loss:      {best_val:.4f} (epoch {best_val_epoch})")
    print(f"Best lesion_err_mm: {best_lesion_err:.3f} mm (epoch {best_lesion_epoch})")
    print(f"Best checkpoint (lesion_err): {ckpt_dir / 'best_by_lesion_err_mm.pth'}")
    print(f"Best checkpoint (val_loss):   {ckpt_dir / 'best_by_val_loss.pth'}")


if __name__ == "__main__":
    main()
