#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Inference script for GMARAFT3D breast biopsy registration.

Old-style output layout:
    out-root/
      inf_YYMMDD_HHMMSS/
        10744914_20210927101351_20210927_rechts/
          warped_moving.nii.gz
          flow_xyz.nii.gz
        10781279_20200316104807_20200316_links/
          ...

Expected CSV columns:
    moving,fixed
Optional metadata columns used for folder naming:
    patient_id, study_id, breast_side
"""

__author__ = "Semih Tarik Uenal"

import csv
import argparse
from datetime import datetime
from pathlib import Path
from typing import Tuple, List

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F

from network_3d.model import GMARAFT_Denoiser3D


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_csv_rows(csv_path: Path) -> List[dict]:
    rows = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("moving") and row.get("fixed"):
                rows.append(row)
    if not rows:
        raise RuntimeError(f"No valid rows in CSV: {csv_path}")
    return rows


def create_out_dir(out_root: Path) -> Path:
    ts = datetime.now().strftime("inf_%y%m%d_%H%M%S")
    d = out_root / ts
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_nifti_xyz(path: str):
    nii = nib.load(path)
    vol = nii.get_fdata().astype(np.float32)
    vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)
    vol = np.clip(vol, -10.0, 10.0)
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D volume, got {vol.shape} at {path}")
    return nii, vol


def zscore(vol: np.ndarray) -> np.ndarray:
    m = float(vol.mean())
    s = float(vol.std())
    if s < 1e-5:
        s = 1e-5
    return ((vol - m) / s).astype(np.float32)


def zscore_to_01(vol_z: np.ndarray, zclamp: float) -> np.ndarray:
    c = float(zclamp)
    v = np.clip(vol_z, -c, c)
    v = (v + c) / (2.0 * c)
    return v.astype(np.float32)


def xyz_to_dhw(vol_xyz: np.ndarray) -> np.ndarray:
    return np.transpose(vol_xyz, (2, 1, 0)).copy()


def dhw_to_xyz(vol_dhw: np.ndarray) -> np.ndarray:
    return np.transpose(vol_dhw, (2, 1, 0)).copy()


def resample_1dhw(x_1dhw: torch.Tensor, target_dhw: Tuple[int, int, int]) -> torch.Tensor:
    x = x_1dhw.unsqueeze(0)
    y = F.interpolate(x, size=target_dhw, mode="trilinear", align_corners=False)
    return y.squeeze(0)


def resize_flow_dhw(flow: torch.Tensor, target_dhw: Tuple[int, int, int]) -> torch.Tensor:
    assert flow.ndim == 5 and flow.shape[1] == 3
    _, _, D1, H1, W1 = flow.shape
    D2, H2, W2 = target_dhw

    flow_res = F.interpolate(flow, size=target_dhw, mode="trilinear", align_corners=False)
    flow_res[:, 0] *= (W2 / W1)
    flow_res[:, 1] *= (H2 / H1)
    flow_res[:, 2] *= (D2 / D1)
    return flow_res


def warp_3d(moving: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    moving = moving.float()
    flow = flow.float()

    B, C, D, H, W = moving.shape

    z = torch.linspace(0, D - 1, D, device=moving.device)
    y = torch.linspace(0, H - 1, H, device=moving.device)
    x = torch.linspace(0, W - 1, W, device=moving.device)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")

    base = torch.stack((xx, yy, zz), dim=-1)[None].repeat(B, 1, 1, 1, 1)
    coords = base + flow.permute(0, 2, 3, 4, 1)

    coords[..., 0] = 2.0 * coords[..., 0] / max(W - 1, 1) - 1.0
    coords[..., 1] = 2.0 * coords[..., 1] / max(H - 1, 1) - 1.0
    coords[..., 2] = 2.0 * coords[..., 2] / max(D - 1, 1) - 1.0

    return F.grid_sample(
        moving,
        coords,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )


def save_nifti_like(ref_nii, data_xyz: np.ndarray, out_path: Path):
    out = nib.Nifti1Image(data_xyz.astype(np.float32), affine=ref_nii.affine, header=ref_nii.header)
    nib.save(out, str(out_path))


def parse_args():
    p = argparse.ArgumentParser("Inference for GMARAFT3D breast biopsy registration")
    p.add_argument("--csv", type=str, required=True, help="CSV with columns moving,fixed")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pth")
    p.add_argument("--out-root", type=str, required=True, help="Output root directory")
    p.add_argument("--input-shape", type=int, nargs=3, default=[224, 224, 96], help="Original XYZ shape")
    p.add_argument("--train-shape", type=int, nargs=3, default=[96, 96, 64], help="Training XYZ shape")
    p.add_argument("--zclamp", type=float, default=5.0)
    p.add_argument("--flow-clamp", type=float, default=20.0)
    p.add_argument("--save-flow", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. This inference script expects a GPU.")

    device = torch.device("cuda")

    csv_path = Path(args.csv).resolve()
    ckpt_path = Path(args.checkpoint).resolve()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    inf_dir = create_out_dir(out_root)

    input_shape_xyz = tuple(args.input_shape)
    train_shape_xyz = tuple(args.train_shape)

    input_dhw = (input_shape_xyz[2], input_shape_xyz[1], input_shape_xyz[0])
    train_dhw = (train_shape_xyz[2], train_shape_xyz[1], train_shape_xyz[0])

    rows = load_csv_rows(csv_path)

    model = GMARAFT_Denoiser3D().to(device)
    state = torch.load(str(ckpt_path), map_location=device)
    if isinstance(state, dict) and "model_state" in state:
        model.load_state_dict(state["model_state"], strict=True)
    else:
        model.load_state_dict(state, strict=True)
    model.eval()

    print("==== GMARAFT3D Inference ====")
    print("CSV          :", csv_path)
    print("Checkpoint   :", ckpt_path)
    print("Out root     :", out_root)
    print("Inference dir:", inf_dir)
    print("Input XYZ    :", input_shape_xyz)
    print("Train XYZ    :", train_shape_xyz)
    print("Cases        :", len(rows))
    print("=============================")

    pred_rows = []

    with torch.no_grad():
        for idx, row in enumerate(rows, start=1):
            moving_path = row["moving"]
            fixed_path = row["fixed"]

            fixed_nii, fixed_xyz = load_nifti_xyz(fixed_path)
            _, moving_xyz = load_nifti_xyz(moving_path)

            if fixed_xyz.shape != input_shape_xyz:
                raise ValueError(f"Fixed shape {fixed_xyz.shape} != expected {input_shape_xyz} at {fixed_path}")
            if moving_xyz.shape != input_shape_xyz:
                raise ValueError(f"Moving shape {moving_xyz.shape} != expected {input_shape_xyz} at {moving_path}")

            fixed_z_xyz = zscore(fixed_xyz)
            moving_z_xyz = zscore(moving_xyz)

            fixed_in_xyz = zscore_to_01(fixed_z_xyz, args.zclamp)
            moving_in_xyz = zscore_to_01(moving_z_xyz, args.zclamp)

            fixed_in_dhw = xyz_to_dhw(fixed_in_xyz)
            moving_in_dhw = xyz_to_dhw(moving_in_xyz)
            moving_z_dhw = xyz_to_dhw(moving_z_xyz)

            fixed_in_t = torch.from_numpy(fixed_in_dhw).unsqueeze(0)
            moving_in_t = torch.from_numpy(moving_in_dhw).unsqueeze(0)
            moving_z_t = torch.from_numpy(moving_z_dhw).unsqueeze(0)

            if train_shape_xyz != input_shape_xyz:
                fixed_in_t = resample_1dhw(fixed_in_t, train_dhw)
                moving_in_t = resample_1dhw(moving_in_t, train_dhw)

            fixed_in_t = fixed_in_t.unsqueeze(0).to(device, non_blocking=True)
            moving_in_t = moving_in_t.unsqueeze(0).to(device, non_blocking=True)

            flows = model(fixed_in_t, moving_in_t)
            flow = flows[-1].float().clamp(-args.flow_clamp, args.flow_clamp)

            if train_shape_xyz != input_shape_xyz:
                flow_full = resize_flow_dhw(flow, input_dhw)
            else:
                flow_full = flow

            moving_z_full = moving_z_t.unsqueeze(0).to(device, non_blocking=True)
            warped_z = warp_3d(moving_z_full, flow_full)

            warped_z_dhw = warped_z[0, 0].cpu().numpy()
            flow_full_np = flow_full[0].cpu().numpy()

            warped_z_xyz = dhw_to_xyz(warped_z_dhw)

            flow_xyz = np.stack([
                dhw_to_xyz(flow_full_np[0]),
                dhw_to_xyz(flow_full_np[1]),
                dhw_to_xyz(flow_full_np[2]),
            ], axis=-1)

            patient_id = str(row.get("patient_id", "unknown_patient"))
            study_id = str(row.get("study_id", f"case_{idx:04d}"))
            breast_side = str(row.get("breast_side", "unknown"))

            case_dir = inf_dir / f"{patient_id}_{study_id}_{breast_side}"
            case_dir.mkdir(parents=True, exist_ok=True)

            warped_path = case_dir / "warped_moving.nii.gz"
            save_nifti_like(fixed_nii, warped_z_xyz, warped_path)

            out_row = dict(row)
            out_row["warped_moving"] = str(warped_path)

            if args.save_flow:
                flow_path = case_dir / "flow.nii.gz"
                save_nifti_like(fixed_nii, flow_xyz, flow_path)
                out_row["flow_xyz"] = str(flow_path)

            pred_rows.append(out_row)
            print(f"[{idx:04d}/{len(rows):04d}] done: {case_dir.name}")

    preds_csv = inf_dir / "preds.csv"
    with preds_csv.open("w", newline="") as f:
        fieldnames = sorted({k for row in pred_rows for k in row.keys()})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pred_rows)

    print(f"Wrote: {preds_csv} rows: {len(pred_rows)}")


if __name__ == "__main__":
    main()