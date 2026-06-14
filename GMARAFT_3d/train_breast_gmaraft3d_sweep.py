#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GMARAFT3D sweep training (stable, NaN-robust) for breast biopsy registration.

Key stability fixes vs. earlier version:
- Model inputs are still z-score -> clamp -> [0,1] (as GMARAFT expects internally).
- Similarity loss (NCC/MSE) is computed on z-score volumes (higher variance, more stable).
- Flow is clamped before warping (prevents flow explosion).
- NCC is computed in FP32 (even if AMP is enabled for the forward pass).
- If NCC becomes NaN/Inf for a batch, it falls back to MSE for that batch.
- If the final loss becomes non-finite, the run aborts (so sweeps fail fast instead of wasting epochs).

CSV expected columns: moving,fixed (optionally other metadata columns are ignored)
"""

__author__ = "Semih Tarik Uenal"

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

    # Loss weights
    sim_weight: float = 1.0
    reg_weight: float = 0.05
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
    p = argparse.ArgumentParser("GMARAFT3D sweep training (stable)")

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

    p.add_argument("--sim-weight", type=float, default=1.0)
    p.add_argument("--reg-weight", type=float, default=0.05)
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
        sim_weight=a.sim_weight,
        reg_weight=a.reg_weight,
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

    def __init__(self, csv_path: Path, input_shape_xyz: Tuple[int, int, int], train_shape_xyz: Tuple[int, int, int], zclamp: float):
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


def flow_smoothness_l1(flow: torch.Tensor) -> torch.Tensor:
    # flow: (B,3,D,H,W)
    # Compute mean absolute spatial gradients (FP32)
    flow = flow.float()
    dx = (flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1]).abs().mean()
    dy = (flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :]).abs().mean()
    dz = (flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :]).abs().mean()
    return (dx + dy + dz) / 3.0


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

    return F.grid_sample(moving, coords, mode="bilinear", padding_mode="border", align_corners=True)


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

    sim_loss_fn = GlobalNCCLoss().to(device)

    best_val = float("inf")

    # wandb summary
    wandb.run.summary["num_train_pairs"] = len(train_ds)
    wandb.run.summary["num_val_pairs"] = len(val_ds)
    wandb.run.summary["exp_dir"] = str(exp_dir)

    print("==== GMARAFT3D Sweep Training (stable) ====")
    print("Exp dir :", exp_dir)
    print("Train   :", wc.train_csv, "pairs=", len(train_ds))
    print("Val     :", wc.val_csv, "pairs=", len(val_ds))
    print("Input xyz:", input_shape_xyz, " Train xyz:", train_shape_xyz)
    print("LR:", float(wc.lr), " reg_weight:", float(wc.reg_weight), " gamma:", float(wc.gamma))
    print("zclamp:", float(wc.zclamp), " flow_clamp:", float(wc.flow_clamp), " AMP:", use_amp)
    print("==========================================")

    for epoch in range(1, int(wc.epochs) + 1):
        t0 = time.time()

        # ----------------- TRAIN ----------------- #
        model.train()
        train_sum = 0.0
        train_batches = 0

        for fixed_in, moving_in, fixed_z, moving_z in train_loader:
            fixed_in = fixed_in.to(device, non_blocking=True)     # (B,1,D,H,W) in [0,1]
            moving_in = moving_in.to(device, non_blocking=True)
            fixed_z = fixed_z.to(device, non_blocking=True)       # (B,1,D,H,W) z-score
            moving_z = moving_z.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # forward can use AMP
            with torch.cuda.amp.autocast(enabled=use_amp):
                flow_preds = model(fixed_in, moving_in)

            # loss in FP32 for stability
            loss = 0.0
            T = len(flow_preds)

            for i, flow in enumerate(flow_preds):
                w = float(wc.gamma) ** (T - i - 1)

                # guard against non-finite flow
                if not torch.isfinite(flow).all():
                    raise RuntimeError("Non-finite flow detected (NaN/Inf) before clamp.")

                flow = flow.float().clamp(-float(wc.flow_clamp), float(wc.flow_clamp))

                warped_z = warp_3d(moving_z, flow)
                sim_raw = sim_loss_fn(fixed_z, warped_z)

                # fallback if NCC becomes non-finite
                if not torch.isfinite(sim_raw):
                    sim_raw = F.mse_loss(fixed_z.float(), warped_z.float())

                reg = flow_smoothness_l1(flow)

                loss = loss + w * (float(wc.sim_weight) * sim_raw + float(wc.reg_weight) * reg)

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

        # ----------------- VAL ----------------- #
        model.eval()
        val_sum = 0.0
        val_batches = 0

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
                    # If model outputs NaN, treat as failed run.
                    raise RuntimeError("Non-finite flow detected (NaN/Inf) during validation.")

                flow = flow.float().clamp(-float(wc.flow_clamp), float(wc.flow_clamp))
                warped_z = warp_3d(moving_z, flow)

                sim_raw = sim_loss_fn(fixed_z, warped_z)
                if not torch.isfinite(sim_raw):
                    sim_raw = F.mse_loss(fixed_z.float(), warped_z.float())

                reg = flow_smoothness_l1(flow)
                val_loss = float(wc.sim_weight) * sim_raw + float(wc.reg_weight) * reg

                if not torch.isfinite(val_loss):
                    raise RuntimeError("Non-finite validation loss detected (NaN/Inf).")

                val_sum += float(val_loss.item())
                val_batches += 1

        val_loss = val_sum / max(1, val_batches)

        dt_min = (time.time() - t0) / 60.0
        print(f"Epoch {epoch:03d}/{int(wc.epochs):03d}  train={train_loss:.4f}  val={val_loss:.4f}  time={dt_min:.1f}min")

        wandb.log(
            {
                "loss/train": train_loss,
                "loss/val": val_loss,
                "lr": float(wc.lr),
                "reg_weight": float(wc.reg_weight),
                "gamma": float(wc.gamma),
                "zclamp": float(wc.zclamp),
                "flow_clamp": float(wc.flow_clamp),
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
