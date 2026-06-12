#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_breast_gmaraft3d_LOSS_axial.py

GMARAFT3D sweep training (stable, NaN-robust) for breast biopsy registration.

Adds an "axial" penalty:
- Detects which flow direction (x/y/z) is strongest (by mean |dx|,|dy|,|dz|)
- Penalizes the strongest direction more (differentiable softmax weighting by default)

Enhancements vs. base script:
- Similarity loss options:
  - global_ncc (FP32), local_ncc (FP32), mse, mixed (alpha*NCC + (1-alpha)*MSE)
- Regularization options (all FP32):
  - 1st-derivative smoothness (L1)
  - bending energy (2nd-derivative, abs/mean)
  - optional Jacobian negative-determinant penalty (anti-folding)
  - optional flow magnitude penalty
  - NEW: optional axial penalty (strongest axis penalized)
- Keeps stability guards:
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

    # NEW: axial penalty (strongest flow axis penalized more)
    axial_weight: float = 0.0    # try 1e-4 .. 1e-2
    axial_temp: float = 0.25     # smaller -> more focus on strongest axis (0.1..0.5)
    axial_mode: str = "softmax"  # softmax | max | anisotropy

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
    p = argparse.ArgumentParser("GMARAFT3D sweep training (stable + flexible losses + axial penalty)")

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

    # NEW: axial penalty args
    p.add_argument("--axial-weight", type=float, default=0.0)
    p.add_argument("--axial-temp", type=float, default=0.25)
    p.add_argument("--axial-mode", type=str, default="softmax", choices=["softmax", "max", "anisotropy"])

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
        axial_weight=a.axial_weight,
        axial_temp=a.axial_temp,
        axial_mode=a.axial_mode,
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
      fixed_in, moving_in, fixed_z, moving_z
    all as torch tensors shaped (1, D, H, W)

    - *_in are zscore->clamp->mapped to [0,1] (stable inputs for GMARAFT)
    - *_z  are z-score volumes (used for similarity loss)
    """

    def __init__(self, csv_path: Path, input_shape_xyz: Tuple[int, int, int],
                 train_shape_xyz: Tuple[int, int, int], zclamp: float):
        super().__init__()
        self.csv_path = Path(csv_path)
        self.input_shape_xyz = tuple(input_shape_xyz)
        self.train_shape_xyz = tuple(train_shape_xyz)
        self.zclamp = float(zclamp)
        self.rows: List[dict] = []

        if not self.csv_path.is_file():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

        with self.csv_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("moving") and row.get("fixed"):
                    self.rows.append(row)

        if not self.rows:
            raise RuntimeError(f"No valid rows in {self.csv_path}")

    def __len__(self) -> int:
        return len(self.rows)

    @staticmethod
    def _load_nifti_xyz(path: str) -> np.ndarray:
        vol = nib.load(path).get_fdata().astype(np.float32)
        vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)
        vol = np.clip(vol, -10.0, 10.0)
        if vol.ndim != 3:
            raise ValueError(f"Expected 3D volume, got {vol.shape} at {path}")
        return vol

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

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        mv_path = row["moving"]
        fx_path = row["fixed"]

        mv_xyz = self._load_nifti_xyz(mv_path)
        fx_xyz = self._load_nifti_xyz(fx_path)

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

        # return fixed first (as used by your GMARAFT call)
        return fx_in_t, mv_in_t, fx_z_t, mv_z_t


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
    # 2nd derivatives - FP32 (abs/mean)
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


def axial_penalty(flow: torch.Tensor, mode: str = "softmax", temp: float = 0.25) -> torch.Tensor:
    """
    Penalize the strongest flow axis (x/y/z) more.

    flow: (B,3,D,H,W) channels = (dx,dy,dz) in voxel units.
    We compute comp = [mean|dx|, mean|dy|, mean|dz|].

    mode:
      - "softmax": differentiable emphasis on the strongest axis (recommended)
      - "max":     only penalize max(comp) (less smooth)
      - "anisotropy": penalize imbalance between axes (max deviation from mean)
    """
    f = flow.float()
    comp = f.abs().mean(dim=(0, 2, 3, 4))  # (3,)

    mode = str(mode).lower()
    if mode == "max":
        return comp.max()

    if mode == "anisotropy":
        return (comp - comp.mean()).abs().max()

    # default: softmax
    t = max(float(temp), 1e-6)
    w = torch.softmax(comp / t, dim=0)  # (3,)
    return (w * comp).sum()


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
        # prefer local if provided
        ncc_term = local_ncc(fixed_z, warped_z) if local_ncc is not None else global_ncc(fixed_z, warped_z)
        mse_term = F.mse_loss(fixed_z.float(), warped_z.float())
        a = float(mixed_alpha)
        a = max(0.0, min(1.0, a))
        return a * ncc_term + (1.0 - a) * mse_term

    # fallback
    return global_ncc(fixed_z, warped_z)


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

    train_ds = BreastPairDataset(Path(wc.train_csv), input_shape_xyz, train_shape_xyz, zclamp=float(wc.zclamp))
    val_ds = BreastPairDataset(Path(wc.val_csv), input_shape_xyz, train_shape_xyz, zclamp=float(wc.zclamp))

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

    print("==== GMARAFT3D Sweep Training (stable + flexible losses + axial penalty) ====")
    print("Exp dir   :", exp_dir)
    print("Train     :", wc.train_csv, "pairs=", len(train_ds))
    print("Val       :", wc.val_csv, "pairs=", len(val_ds))
    print("Input xyz :", input_shape_xyz, " Train xyz:", train_shape_xyz)
    print("LR        :", float(wc.lr))
    print("Sim       :", str(wc.sim_type), " sim_weight:", float(wc.sim_weight), " ncc_win:", int(wc.ncc_win), " mixed_alpha:", float(wc.mixed_alpha))
    print("Reg       :", "reg_weight:", float(wc.reg_weight), f" smooth_frac:{smooth_frac:.2f} bend_frac:{bend_frac:.2f}",
          " jac_weight:", float(wc.jac_weight), " mag_weight:", float(wc.mag_weight))
    print("Axial     :", "axial_weight:", float(wc.axial_weight), " axial_mode:", str(wc.axial_mode), " axial_temp:", float(wc.axial_temp))
    print("gamma     :", float(wc.gamma))
    print("zclamp    :", float(wc.zclamp), " flow_clamp:", float(wc.flow_clamp), " AMP:", use_amp)
    print("===========================================================================")

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
        tr_axial_sum = 0.0

        for fixed_in, moving_in, fixed_z, moving_z in train_loader:
            fixed_in = fixed_in.to(device, non_blocking=True)     # (B,1,D,H,W) in [0,1]
            moving_in = moving_in.to(device, non_blocking=True)
            fixed_z = fixed_z.to(device, non_blocking=True)       # (B,1,D,H,W) z-score
            moving_z = moving_z.to(device, non_blocking=True)

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
            axial_weight = float(wc.axial_weight)

            gamma = float(wc.gamma)
            flow_clamp = float(wc.flow_clamp)

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
                axial = axial_penalty(flow, mode=str(wc.axial_mode), temp=float(wc.axial_temp)) if axial_weight > 0.0 else flow.new_tensor(0.0)

                step_loss = (sim_weight * sim_raw) + (reg_weight * reg) + (jac_weight * jac) + (mag_weight * mag) + (axial_weight * axial)
                loss = loss + w * step_loss

                # accumulate components (weighted)
                tr_sim_sum += float((w * sim_raw).item())
                tr_smooth_sum += float((w * smooth).item())
                tr_bend_sum += float((w * bend).item())
                tr_jac_sum += float((w * jac).item())
                tr_mag_sum += float((w * mag).item())
                tr_axial_sum += float((w * axial).item())

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
        tr_axial = tr_axial_sum / max(1, train_batches)

        # ----------------- VAL ----------------- #
        model.eval()
        val_sum = 0.0
        val_batches = 0

        va_sim_sum = 0.0
        va_smooth_sum = 0.0
        va_bend_sum = 0.0
        va_jac_sum = 0.0
        va_mag_sum = 0.0
        va_axial_sum = 0.0

        # track dominant axis counts in validation (0=x,1=y,2=z)
        dom_counts = [0, 0, 0]

        with torch.no_grad():
            for fixed_in, moving_in, fixed_z, moving_z in val_loader:
                fixed_in = fixed_in.to(device, non_blocking=True)
                moving_in = moving_in.to(device, non_blocking=True)
                fixed_z = fixed_z.to(device, non_blocking=True)
                moving_z = moving_z.to(device, non_blocking=True)

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
                axial = axial_penalty(flow, mode=str(wc.axial_mode), temp=float(wc.axial_temp)) if float(wc.axial_weight) > 0.0 else flow.new_tensor(0.0)

                val_loss = (
                    float(wc.sim_weight) * sim_raw
                    + float(wc.reg_weight) * reg
                    + float(wc.jac_weight) * jac
                    + float(wc.mag_weight) * mag
                    + float(wc.axial_weight) * axial
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
                va_axial_sum += float(axial.item())

                # dominant axis stats (based on mean abs component)
                comp = flow.abs().mean(dim=(0, 2, 3, 4))  # (3,)
                dom = int(comp.argmax().item())
                dom_counts[dom] += 1

        val_loss = val_sum / max(1, val_batches)
        va_sim = va_sim_sum / max(1, val_batches)
        va_smooth = va_smooth_sum / max(1, val_batches)
        va_bend = va_bend_sum / max(1, val_batches)
        va_jac = va_jac_sum / max(1, val_batches)
        va_mag = va_mag_sum / max(1, val_batches)
        va_axial = va_axial_sum / max(1, val_batches)

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
                "comp/train_axial_wsum": tr_axial,

                "comp/val_sim": va_sim,
                "comp/val_smooth": va_smooth,
                "comp/val_bend": va_bend,
                "comp/val_jac": va_jac,
                "comp/val_mag": va_mag,
                "comp/val_axial": va_axial,

                "axial/dom_x_frac": dom_counts[0] / max(1, val_batches),
                "axial/dom_y_frac": dom_counts[1] / max(1, val_batches),
                "axial/dom_z_frac": dom_counts[2] / max(1, val_batches),

                "lr": float(wc.lr),
                "reg_weight": float(wc.reg_weight),
                "sim_weight": float(wc.sim_weight),
                "jac_weight": float(wc.jac_weight),
                "mag_weight": float(wc.mag_weight),
                "axial_weight": float(wc.axial_weight),
                "axial_temp": float(wc.axial_temp),
                "axial_mode": str(wc.axial_mode),
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