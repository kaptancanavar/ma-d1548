#!/usr/bin/env python3
# -*- coding: utf-8 -*-


__author__ = "Semih Tarik Uenal"

import csv, time, argparse
from datetime import datetime
from pathlib import Path
from typing import Tuple, List

import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from network_3d.model import GMARAFT_Denoiser3D


# ---------------- Dataset (CSV: moving,fixed) ---------------- #

class BreastPairDataset(Dataset):
    """
    CSV must contain columns: moving,fixed
    Loaded NIfTI assumed shape: (X,Y,Z) = (224,224,96)
    GMARAFT3D expects torch shape: (B,1,D,H,W) with D=Z, H=Y, W=X.
    """

    def __init__(self, csv_path: Path, input_shape_xyz: Tuple[int,int,int], train_shape_xyz: Tuple[int,int,int]):
        self.csv_path = Path(csv_path)
        self.input_shape_xyz = tuple(input_shape_xyz)
        self.train_shape_xyz = tuple(train_shape_xyz)
        self.rows: List[dict] = []

        with self.csv_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("moving") and row.get("fixed"):
                    self.rows.append(row)
        if not self.rows:
            raise RuntimeError(f"No valid rows in {self.csv_path}")

    @staticmethod
    def _load_nifti(path: str) -> np.ndarray:
        vol = nib.load(path).get_fdata().astype(np.float32)
        vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)
        vol = np.clip(vol, -10.0, 10.0)
        # per-volume z-score (wie bei dir)
        m = float(vol.mean())
        s = float(vol.std())
        if s < 1e-5:
            s = 1e-5
        vol = (vol - m) / s
        return vol

    @staticmethod
    def _zscore_to_01(vol: np.ndarray, clamp: float = 5.0) -> np.ndarray:
        # map z-score roughly into [0,1], because model does x->2x-1 internally
        vol = np.clip(vol, -clamp, clamp)
        vol = (vol + clamp) / (2.0 * clamp)
        return vol.astype(np.float32)

    @staticmethod
    def _xyz_to_dhw(vol_xyz: np.ndarray) -> np.ndarray:
        # (X,Y,Z) -> (D,H,W) = (Z,Y,X)
        return np.transpose(vol_xyz, (2, 1, 0)).copy()

    @staticmethod
    def _resample_dhw(x_1dhw: torch.Tensor, target_dhw: Tuple[int,int,int]) -> torch.Tensor:
        # x_1dhw: (1,D,H,W) -> (1,1,D,H,W) -> interpolate -> (1,D,H,W)
        x = x_1dhw.unsqueeze(0)
        y = F.interpolate(x, size=target_dhw, mode="trilinear", align_corners=False)
        return y.squeeze(0)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        mv_path = row["moving"]
        fx_path = row["fixed"]

        mv = self._load_nifti(mv_path)
        fx = self._load_nifti(fx_path)

        if mv.shape != self.input_shape_xyz or fx.shape != self.input_shape_xyz:
            raise ValueError(f"Expected {self.input_shape_xyz} but got mv={mv.shape}, fx={fx.shape}")

        mv = self._zscore_to_01(mv)
        fx = self._zscore_to_01(fx)

        mv = self._xyz_to_dhw(mv)  # (D,H,W)
        fx = self._xyz_to_dhw(fx)

        mv_t = torch.from_numpy(mv).unsqueeze(0)  # (1,D,H,W)
        fx_t = torch.from_numpy(fx).unsqueeze(0)

        # train resolution (smaller!)
        if self.train_shape_xyz != self.input_shape_xyz:
            train_dhw = (self.train_shape_xyz[2], self.train_shape_xyz[1], self.train_shape_xyz[0])
            mv_t = self._resample_dhw(mv_t, train_dhw)
            fx_t = self._resample_dhw(fx_t, train_dhw)

        return fx_t, mv_t  # fixed, moving


# ---------------- Losses / Warp ---------------- #

class GlobalNCCLoss(nn.Module):
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x,y: (B,1,D,H,W)
        x = x[:, 0].reshape(x.shape[0], -1)
        y = y[:, 0].reshape(y.shape[0], -1)
        x = x - x.mean(dim=1, keepdim=True)
        y = y - y.mean(dim=1, keepdim=True)
        varx = (x*x).mean(dim=1, keepdim=True)
        vary = (y*y).mean(dim=1, keepdim=True)
        denom = torch.sqrt(varx * vary).clamp_min(self.eps)
        ncc = (x*y).mean(dim=1, keepdim=True) / denom
        return -ncc.mean()

def flow_smoothness_l1(flow: torch.Tensor) -> torch.Tensor:
    # flow: (B,3,D,H,W)
    dx = (flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1]).abs().mean()
    dy = (flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :]).abs().mean()
    dz = (flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :]).abs().mean()
    return (dx + dy + dz) / 3.0

def warp_3d(moving: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """
    moving: (B,1,D,H,W)
    flow:   (B,3,D,H,W) with channels (dx,dy,dz) in voxel units of (W,H,D)
    """
    B, C, D, H, W = moving.shape
    z = torch.linspace(0, D - 1, D, device=moving.device)
    y = torch.linspace(0, H - 1, H, device=moving.device)
    x = torch.linspace(0, W - 1, W, device=moving.device)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    base = torch.stack((xx, yy, zz), dim=-1)[None].repeat(B, 1, 1, 1, 1)  # (B,D,H,W,3)

    coords = base + flow.permute(0, 2, 3, 4, 1)  # add (dx,dy,dz)

    coords[..., 0] = 2.0 * coords[..., 0] / max(W - 1, 1) - 1.0
    coords[..., 1] = 2.0 * coords[..., 1] / max(H - 1, 1) - 1.0
    coords[..., 2] = 2.0 * coords[..., 2] / max(D - 1, 1) - 1.0

    return F.grid_sample(moving, coords, mode="bilinear", padding_mode="border", align_corners=True)

def upsample_flow(flow: torch.Tensor, target_dhw: Tuple[int,int,int]) -> torch.Tensor:
    # flow in voxel units at its own resolution -> upsample + scale components
    D1, H1, W1 = flow.shape[-3:]
    D2, H2, W2 = target_dhw
    up = F.interpolate(flow, size=(D2, H2, W2), mode="trilinear", align_corners=False)
    up[:, 0] *= (W2 / float(W1))  # dx
    up[:, 1] *= (H2 / float(H1))  # dy
    up[:, 2] *= (D2 / float(D1))  # dz
    return up


# ---------------- Train ---------------- #

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-csv", type=str, required=True)
    p.add_argument("--val-csv", type=str, required=True)
    p.add_argument("--out-root", type=str, default="experiments_gmaraft")

    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)

    # given as (X,Y,Z) like your VoxelMorph script
    p.add_argument("--input-shape", type=int, nargs=3, default=[224,224,96])
    p.add_argument("--train-shape", type=int, nargs=3, default=[96,96,64])

    p.add_argument("--sim-weight", type=float, default=1.0)
    p.add_argument("--reg-weight", type=float, default=0.01)
    p.add_argument("--gamma", type=float, default=0.85)
    p.add_argument("--save-every", type=int, default=10)
    return p.parse_args()

def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("GMARAFT3D in this repo uses CUDA autocast internally. Run on a GPU node.")

    device = torch.device("cuda")
    root = Path(__file__).resolve().parent
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    exp_dir = out_root / datetime.now().strftime("run_%y%m%d_%H%M%S")
    ckpt_dir = exp_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    input_shape_xyz = tuple(args.input_shape)
    train_shape_xyz = tuple(args.train_shape)

    train_ds = BreastPairDataset(Path(args.train_csv), input_shape_xyz, train_shape_xyz)
    val_ds   = BreastPairDataset(Path(args.val_csv),   input_shape_xyz, train_shape_xyz)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    model = GMARAFT_Denoiser3D().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-5)
    scaler = torch.cuda.amp.GradScaler(True)
    ncc = GlobalNCCLoss().to(device)

    best_val = float("inf")

    # target full-res in DHW for upsample/warp (D=Z,H=Y,W=X)
    full_dhw = (input_shape_xyz[2], input_shape_xyz[1], input_shape_xyz[0])

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # ----- TRAIN -----
        model.train()
        train_sum = 0.0
        for fixed, moving in train_loader:
            fixed = fixed.to(device, non_blocking=True)     # (B,1,D,H,W) at train_dhw
            moving = moving.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(True):
                flows = model(fixed, moving)  # list of flows (B,3,D,H,W)
                loss = 0.0
                T = len(flows)
                for i, flow in enumerate(flows):
                    w = args.gamma ** (T - i - 1)
                    warped = warp_3d(moving, flow)
                    sim = ncc(fixed, warped)
                    reg = flow_smoothness_l1(flow)
                    loss = loss + w * (args.sim_weight * sim + args.reg_weight * reg)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            train_sum += float(loss.item())
        train_loss = train_sum / max(1, len(train_loader))

        # ----- VAL -----
        model.eval()
        val_sum = 0.0
        with torch.no_grad():
            for fixed, moving in val_loader:
                fixed = fixed.to(device, non_blocking=True)
                moving = moving.to(device, non_blocking=True)

                flows = model(fixed, moving)
                flow = flows[-1]
                warped = warp_3d(moving, flow)
                sim = ncc(fixed, warped)
                reg = flow_smoothness_l1(flow)
                val_sum += float((args.sim_weight * sim + args.reg_weight * reg).item())
        val_loss = val_sum / max(1, len(val_loader))

        dt_min = (time.time() - t0) / 60.0
        print(f"Epoch {epoch:03d}/{args.epochs}  train={train_loss:.4f}  val={val_loss:.4f}  time={dt_min:.1f}min")

        # best + periodic ckpts
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), ckpt_dir / "best_model.pth")
        if (epoch % args.save_every == 0) or (epoch == args.epochs):
            torch.save(model.state_dict(), ckpt_dir / f"epoch_{epoch:03d}.pth")

    print("Done. Best val:", best_val)
    print("Exp dir:", exp_dir)

if __name__ == "__main__":
    main()
