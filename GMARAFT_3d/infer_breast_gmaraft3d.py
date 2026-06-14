#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Inference script for GMARAFT3D on MR breast biopsy data.

- CSV columns: moving,fixed (+ optional patient_id, study_id, breast_side)
- Expects input volumes in shape (X,Y,Z) e.g. (224,224,96)
- Model expects torch shape (B,1,D,H,W) where D=Z, H=Y, W=X

Outputs per pair:
- warped_moving.nii.gz  (moving warped into fixed space)
- optional flow.nii.gz  (4D, shape (X,Y,Z,3), components = (dx,dy,dz) in voxels)
"""

__author__ = "Semih Tarik Uenal"

import os
import csv
import argparse
from datetime import datetime
from pathlib import Path
from typing import Tuple, List

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from network_3d.model import GMARAFT_Denoiser3D


# ---------------- Dataset ---------------- #

class BreastPairInferenceDataset(Dataset):
    def __init__(self, csv_path: Path, input_shape_xyz: Tuple[int, int, int]):
        super().__init__()
        self.csv_path = Path(csv_path)
        self.input_shape_xyz = tuple(input_shape_xyz)
        self.rows: List[dict] = []

        if not self.csv_path.is_file():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

        with self.csv_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("moving") and row.get("fixed"):
                    self.rows.append(row)

        if not self.rows:
            raise RuntimeError(f"No valid rows found in {self.csv_path}")

        print(f"[InferenceDataset] {self.csv_path} -> {len(self.rows)} pairs")

    def __len__(self):
        return len(self.rows)

    @staticmethod
    def _load_nifti_xyz(path: str) -> np.ndarray:
        vol = nib.load(path).get_fdata().astype(np.float32)
        vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)
        vol = np.clip(vol, -10.0, 10.0)
        if vol.ndim != 3:
            raise ValueError(f"Expected 3D volume, got shape {vol.shape} at {path}")
        return vol

    @staticmethod
    def _zscore(vol: np.ndarray) -> np.ndarray:
        m = float(vol.mean())
        s = float(vol.std())
        if s < 1e-5:
            s = 1e-5
        return ((vol - m) / s).astype(np.float32)

    def __getitem__(self, idx):
        row = self.rows[idx]
        mv_path = row["moving"]
        fx_path = row["fixed"]

        mv_xyz = self._load_nifti_xyz(mv_path)
        fx_xyz = self._load_nifti_xyz(fx_path)

        if mv_xyz.shape != self.input_shape_xyz:
            raise ValueError(f"Moving shape {mv_xyz.shape} != expected {self.input_shape_xyz} at {mv_path}")
        if fx_xyz.shape != self.input_shape_xyz:
            raise ValueError(f"Fixed shape {fx_xyz.shape} != expected {self.input_shape_xyz} at {fx_path}")

        # match training: per-volume z-score
        mv_z_xyz = self._zscore(mv_xyz)
        fx_z_xyz = self._zscore(fx_xyz)

        meta = {
            "mv_path": mv_path,
            "fx_path": fx_path,
            "patient_id": row.get("patient_id", "unknown"),
            "study_id": row.get("study_id", "unknown"),
            "breast_side": row.get("breast_side", "unknown"),
        }

        # return as numpy; conversion to torch later
        return mv_z_xyz, fx_z_xyz, meta


# ---------------- Utils ---------------- #

def xyz_to_dhw(vol_xyz: np.ndarray) -> np.ndarray:
    # (X,Y,Z) -> (D,H,W) = (Z,Y,X)
    return np.transpose(vol_xyz, (2, 1, 0)).copy()

def dhw_to_xyz(vol_dhw: np.ndarray) -> np.ndarray:
    # (D,H,W) -> (X,Y,Z)
    return np.transpose(vol_dhw, (2, 1, 0)).copy()

def zscore_to_01(vol_z: np.ndarray, clamp: float = 5.0) -> np.ndarray:
    # clamp z-score and map to [0,1] for GMARAFT input
    v = np.clip(vol_z, -clamp, clamp)
    v = (v + clamp) / (2.0 * clamp)
    return v.astype(np.float32)

def resample_1dhw(x_1dhw: torch.Tensor, target_dhw: Tuple[int, int, int]) -> torch.Tensor:
    # x_1dhw: (1,D,H,W)
    x = x_1dhw.unsqueeze(0)  # (1,1,D,H,W)
    y = F.interpolate(x, size=target_dhw, mode="trilinear", align_corners=False)
    return y.squeeze(0)      # (1,D,H,W)

def upsample_flow(flow: torch.Tensor, target_dhw: Tuple[int, int, int]) -> torch.Tensor:
    """
    flow: (B,3,D,H,W) at current resolution (D,H,W)
    target_dhw: (D2,H2,W2)
    returns: flow upsampled + scaled in voxel units
    """
    D1, H1, W1 = flow.shape[-3:]
    D2, H2, W2 = target_dhw
    up = F.interpolate(flow, size=(D2, H2, W2), mode="trilinear", align_corners=False)
    up[:, 0] *= (W2 / float(W1))  # dx
    up[:, 1] *= (H2 / float(H1))  # dy
    up[:, 2] *= (D2 / float(D1))  # dz
    return up

def warp_3d(moving: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """
    moving: (B,1,D,H,W)
    flow:   (B,3,D,H,W) with channels (dx,dy,dz) in voxel units (x=W, y=H, z=D)
    """
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

    return F.grid_sample(moving, coords, mode="bilinear", padding_mode="border", align_corners=True)

def create_out_dir(out_root: Path) -> Path:
    ts = datetime.now().strftime("inf_%y%m%d_%H%M%S")
    d = out_root / ts
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------- Main ---------------- #

def parse_args():
    p = argparse.ArgumentParser("GMARAFT3D inference (CSV moving/fixed, NIfTI)")
    p.add_argument("--csv", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--out-root", type=str, required=True)

    # shapes are (X,Y,Z)
    p.add_argument("--input-shape", type=int, nargs=3, default=[224, 224, 96])
    p.add_argument("--train-shape", type=int, nargs=3, default=[96, 96, 64],
                   help="Shape used during training/inference for the model. Flow is upsampled to input-shape.")

    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--save-flow", action="store_true")
    p.add_argument("--zclamp", type=float, default=5.0, help="Clamp for z-score -> [0,1] mapping")

    return p.parse_args()

def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. GMARAFT3D inference expects a GPU node.")

    device = torch.device("cuda")

    csv_path = Path(args.csv).resolve()
    ckpt_path = Path(args.checkpoint).resolve()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    input_shape_xyz = tuple(args.input_shape)
    train_shape_xyz = tuple(args.train_shape)

    # full + train shapes in DHW
    full_dhw = (input_shape_xyz[2], input_shape_xyz[1], input_shape_xyz[0])
    train_dhw = (train_shape_xyz[2], train_shape_xyz[1], train_shape_xyz[0])

    print("==== GMARAFT3D Inference ====")
    print("CSV       :", csv_path)
    print("Checkpoint:", ckpt_path)
    print("Out root  :", out_root)
    print("Input xyz :", input_shape_xyz, "-> full dhw:", full_dhw)
    print("Train xyz :", train_shape_xyz, "-> train dhw:", train_dhw)
    print("-----------------------------")

    ds = BreastPairInferenceDataset(csv_path, input_shape_xyz=input_shape_xyz)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    model = GMARAFT_Denoiser3D().to(device)

    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "model_state" in state:
        model.load_state_dict(state["model_state"])
        print(f"Loaded checkpoint dict (epoch={state.get('epoch', 'unknown')})")
    else:
        model.load_state_dict(state)
        print("Loaded model state_dict.")

    model.eval()

    inf_dir = create_out_dir(out_root)
    print("Inference dir:", inf_dir)

    with torch.no_grad():
        for batch_idx, (mv_z_xyz, fx_z_xyz, meta) in enumerate(loader):
            # DataLoader makes mv_z_xyz, fx_z_xyz numpy->object? ensure tensors via numpy conversion
            # They come as tensors only if collate can handle numpy; safest: convert explicitly.

            B = len(meta["mv_path"])

            for b in range(B):
                mv_path = meta["mv_path"][b]
                fx_path = meta["fx_path"][b]
                patient_id = meta.get("patient_id", ["unknown"] * B)[b]
                study_id = meta.get("study_id", ["unknown"] * B)[b]
                breast_side = meta.get("breast_side", ["unknown"] * B)[b]

                # mv_z_xyz, fx_z_xyz are batches of arrays; convert for this element
                mv_z = np.asarray(mv_z_xyz[b], dtype=np.float32)
                fx_z = np.asarray(fx_z_xyz[b], dtype=np.float32)

                # create model inputs: zscore -> [0,1]
                mv_01 = zscore_to_01(mv_z, clamp=args.zclamp)
                fx_01 = zscore_to_01(fx_z, clamp=args.zclamp)

                # xyz -> dhw
                mv_01_dhw = xyz_to_dhw(mv_01)
                fx_01_dhw = xyz_to_dhw(fx_01)

                mv_z_dhw = xyz_to_dhw(mv_z)  # for final warp output (zscore domain)

                # to torch (1,D,H,W)
                mv_in = torch.from_numpy(mv_01_dhw).unsqueeze(0)
                fx_in = torch.from_numpy(fx_01_dhw).unsqueeze(0)

                # resample to train_dhw if needed
                if train_dhw != full_dhw:
                    mv_in = resample_1dhw(mv_in, train_dhw)
                    fx_in = resample_1dhw(fx_in, train_dhw)

                # add batch -> (1,1,D,H,W)
                mv_in = mv_in.unsqueeze(0).to(device, non_blocking=True)
                fx_in = fx_in.unsqueeze(0).to(device, non_blocking=True)

                # forward -> list of flow predictions, take last
                flows = model(fx_in, mv_in)
                flow = flows[-1]  # (1,3,D,H,W) at train_dhw

                # upsample flow to full resolution if needed
                if train_dhw != full_dhw:
                    flow_full = upsample_flow(flow, full_dhw)
                else:
                    flow_full = flow

                # warp FULL-RES moving in z-score domain
                mv_z_t = torch.from_numpy(mv_z_dhw).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,D,H,W)
                warped = warp_3d(mv_z_t, flow_full)  # (1,1,D,H,W)

                warped_dhw = warped[0, 0].detach().cpu().numpy().astype(np.float32)
                warped_xyz = dhw_to_xyz(warped_dhw)

                # save with fixed affine/header
                fx_img = nib.load(fx_path)
                fx_aff = fx_img.affine

                case_dir = inf_dir / f"{patient_id}_{study_id}_{breast_side}"
                case_dir.mkdir(parents=True, exist_ok=True)

                warped_path = case_dir / "warped_moving.nii.gz"
                nib.save(nib.Nifti1Image(warped_xyz, fx_aff, header=fx_img.header), str(warped_path))

                if args.save_flow:
                    # flow_full: (1,3,D,H,W) -> (D,H,W,3) -> (X,Y,Z,3)
                    flow_dhw_3 = flow_full[0].permute(1, 2, 3, 0).detach().cpu().numpy().astype(np.float32)  # (D,H,W,3)
                    flow_xyz_3 = np.transpose(flow_dhw_3, (2, 1, 0, 3))  # (X,Y,Z,3)
                    flow_path = case_dir / "flow.nii.gz"
                    nib.save(nib.Nifti1Image(flow_xyz_3, fx_aff, header=fx_img.header), str(flow_path))

                print(f"[{batch_idx:04d}:{b}] Saved to {case_dir} (flow={args.save_flow})")

    print("Inference finished.")
    print("All outputs under:", inf_dir)


if __name__ == "__main__":
    main()
