#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train TransMorph on preprocessed MR breast biopsy data using the same CSV setup
as the user's VoxelMorph training script.

CSV columns required:
  moving,fixed

Optional extra columns are ignored.

Expected data:
  - NIfTI volumes
  - already preprocessed
  - default shape: (224, 224, 96)

Outputs:
  --out-root/run_<timestamp>/
    config.txt
    logs/
    checkpoints/
      checkpoint_epoch_XXX.pth
      best_model.pth
"""

__author__ = "Semih Tarik Uenal"

import os
import sys
import csv
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Tuple, List

import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import wandb


# ---------------- Dataset ---------------- #

class BreastPairDataset(Dataset):
    def __init__(self, csv_path: Path, target_shape: Tuple[int, int, int]):
        super().__init__()
        self.csv_path = Path(csv_path)
        self.target_shape = tuple(target_shape)
        self.rows: List[dict] = []

        if not self.csv_path.is_file():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

        with self.csv_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                mv = row.get("moving")
                fx = row.get("fixed")
                if not mv or not fx:
                    continue
                self.rows.append(row)

        if not self.rows:
            raise RuntimeError(f"No valid rows found in {self.csv_path}")

        print(f"[Dataset] {self.csv_path} -> {len(self.rows)} pairs")

    def __len__(self) -> int:
        return len(self.rows)

    @staticmethod
    def _load_nifti(path: str) -> np.ndarray:
        img = nib.load(path)
        data = img.get_fdata().astype(np.float32)

        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        data = np.clip(data, -10.0, 10.0)

        if data.ndim != 3:
            raise ValueError(f"Expected 3D volume, got shape {data.shape} at {path}")
        return data

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        mv_path = row["moving"]
        fx_path = row["fixed"]

        mv_vol = self._load_nifti(mv_path)
        fx_vol = self._load_nifti(fx_path)

        if mv_vol.shape != self.target_shape:
            raise ValueError(
                f"Moving volume shape {mv_vol.shape} != target_shape {self.target_shape} at {mv_path}"
            )
        if fx_vol.shape != self.target_shape:
            raise ValueError(
                f"Fixed volume shape {fx_vol.shape} != target_shape {self.target_shape} at {fx_path}"
            )

        mv = torch.from_numpy(mv_vol).unsqueeze(0)  # (1, X, Y, Z)
        fx = torch.from_numpy(fx_vol).unsqueeze(0)
        return mv, fx


# ---------------- Losses ---------------- #

class GlobalNCCLoss(nn.Module):
    """
    Global normalized cross-correlation.
    Returns loss = -NCC.
    """
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


class SoftMILoss(nn.Module):
    """
    Differentiable soft mutual information.
    Returns loss = -MI.
    """
    def __init__(self, bins=32, sigma=0.2, samples=20000, clip=5.0, eps=1e-6):
        super().__init__()
        self.bins = int(bins)
        self.sigma = float(sigma)
        self.samples = int(samples)
        self.clip = float(clip)
        self.eps = float(eps)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.ndim == 5 and x.shape[1] == 1:
            x = x[:, 0, ...]
            y = y[:, 0, ...]

        x = x.reshape(x.shape[0], -1).float()
        y = y.reshape(y.shape[0], -1).float()

        N = x.shape[1]
        if self.samples > 0 and self.samples < N:
            idx = torch.randperm(N, device=x.device)[: self.samples]
            x = x[:, idx]
            y = y[:, idx]

        x = torch.clamp(x, -self.clip, self.clip)
        y = torch.clamp(y, -self.clip, self.clip)

        centers = torch.linspace(-self.clip, self.clip, self.bins, device=x.device).view(1, 1, -1)

        sigma = max(self.sigma, 1e-4)
        xw = torch.exp(-0.5 * ((x.unsqueeze(-1) - centers) / sigma) ** 2)
        yw = torch.exp(-0.5 * ((y.unsqueeze(-1) - centers) / sigma) ** 2)

        xw = xw / (xw.sum(dim=-1, keepdim=True) + self.eps)
        yw = yw / (yw.sum(dim=-1, keepdim=True) + self.eps)

        pxy = torch.bmm(xw.transpose(1, 2), yw) / xw.shape[1]
        px = pxy.sum(dim=2, keepdim=True)
        py = pxy.sum(dim=1, keepdim=True)

        mi = pxy * (torch.log(pxy + self.eps) - torch.log(px * py + self.eps))
        mi = mi.sum(dim=(1, 2))
        return -mi.mean()


class Grad3dLoss(nn.Module):
    """
    Smoothness loss on displacement field.
    Similar role as VoxelMorph Grad loss.
    """
    def __init__(self, penalty: str = "l2", eps: float = 1e-8):
        super().__init__()
        if penalty not in ("l1", "l2"):
            raise ValueError("penalty must be 'l1' or 'l2'")
        self.penalty = penalty
        self.eps = eps

    def forward(self, flow: torch.Tensor) -> torch.Tensor:
        dy = flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :]
        dx = flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :]
        dz = flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1]

        if self.penalty == "l2":
            dx = dx * dx
            dy = dy * dy
            dz = dz * dz
        else:
            dx = torch.abs(dx)
            dy = torch.abs(dy)
            dz = torch.abs(dz)

        return (dx.mean() + dy.mean() + dz.mean()) / 3.0


class AxialFlowLoss(nn.Module):
    """
    Penalizes displacement magnitude per axis.
    Useful when x/y/z motion should be weighted differently.
    Expects flow shape: (B, 3, X, Y, Z)
    """
    def __init__(
        self,
        wx: float = 1.0,
        wy: float = 1.0,
        wz: float = 1.0,
        penalty: str = "l2",
    ):
        super().__init__()
        if penalty not in ("l1", "l2"):
            raise ValueError("penalty must be 'l1' or 'l2'")
        self.wx = float(wx)
        self.wy = float(wy)
        self.wz = float(wz)
        self.penalty = penalty

    def forward(self, flow: torch.Tensor) -> torch.Tensor:
        if flow.ndim != 5 or flow.shape[1] != 3:
            raise ValueError(f"Expected flow shape (B, 3, X, Y, Z), got {tuple(flow.shape)}")

        fx = flow[:, 0, ...]
        fy = flow[:, 1, ...]
        fz = flow[:, 2, ...]

        if self.penalty == "l2":
            lx = (fx * fx).mean()
            ly = (fy * fy).mean()
            lz = (fz * fz).mean()
        else:
            lx = torch.abs(fx).mean()
            ly = torch.abs(fy).mean()
            lz = torch.abs(fz).mean()

        denom = self.wx + self.wy + self.wz + 1e-8
        return (self.wx * lx + self.wy * ly + self.wz * lz) / denom


# ---------------- TransMorph import ---------------- #

def import_transmorph(repo_root: Path):
    repo_root = Path(repo_root).resolve()
    tm_root = repo_root / "TransMorph"

    if not repo_root.is_dir():
        raise FileNotFoundError(f"TransMorph repo root not found: {repo_root}")
    if not tm_root.is_dir():
        raise FileNotFoundError(f"Expected folder not found: {tm_root}")

    sys.path.insert(0, str(tm_root))

    from models.TransMorph import TransMorph  # type: ignore
    import models.configs_TransMorph as configs  # type: ignore

    return TransMorph, configs


def get_config_by_variant(configs_module, variant: str):
    variant = variant.lower()

    candidates = {
        "tiny": [
            "get_3DTransMorphTiny_config",
            "get_3DTransMorph_tiny_config",
        ],
        "small": [
            "get_3DTransMorphSmall_config",
            "get_3DTransMorph_small_config",
        ],
        "base": [
            "get_3DTransMorph_config",
            "get_3DTransMorphBase_config",
            "get_3DTransMorph_base_config",
        ],
    }

    if variant not in candidates:
        raise ValueError(f"Unknown TransMorph variant: {variant}")

    for fn_name in candidates[variant]:
        if hasattr(configs_module, fn_name):
            cfg = getattr(configs_module, fn_name)()
            return cfg, fn_name

    available = [x for x in dir(configs_module) if x.startswith("get_")]
    raise RuntimeError(
        f"Could not find config function for variant '{variant}'. "
        f"Available config builders: {available}"
    )


# ---------------- Utility ---------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train TransMorph on breast biopsy MRIs.")

    # paths
    p.add_argument("--transmorph-repo", type=str, required=True,
                   help="Path to official TransMorph repo root.")
    p.add_argument("--train-csv", type=str, required=True)
    p.add_argument("--val-csv", type=str, required=True)
    p.add_argument("--out-root", type=str, required=True)

    # training
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)

    p.add_argument("--image-loss", type=str, default="ncc", choices=["mse", "ncc", "mi"])
    p.add_argument("--sim-weight", type=float, default=1.0)
    p.add_argument("--reg-weight", type=float, default=0.1)
    p.add_argument("--axial-weight", type=float, default=0.0,
                   help="Weight for additional axis-aware flow magnitude penalty.")
    p.add_argument("--axial-x", type=float, default=1.0,
                   help="Penalty weight for x displacement component.")
    p.add_argument("--axial-y", type=float, default=1.0,
                   help="Penalty weight for y displacement component.")
    p.add_argument("--axial-z", type=float, default=1.0,
                   help="Penalty weight for z displacement component.")
    p.add_argument("--axial-penalty", type=str, default="l2", choices=["l1", "l2"],
                   help="Penalty type for axial flow loss.")
    p.add_argument("--save-every", type=int, default=10)

    p.add_argument("--input-shape", type=int, nargs=3, default=[224, 224, 96], metavar=("NX", "NY", "NZ"))
    p.add_argument("--window-size", type=int, nargs=3, default=[7, 7, 3], metavar=("WX", "WY", "WZ"))

    p.add_argument("--transmorph-variant", type=str, default="small", choices=["tiny", "small", "base"])
    p.add_argument("--exp-id", type=str, default="")

    # MI
    p.add_argument("--mi-bins", type=int, default=32)
    p.add_argument("--mi-sigma", type=float, default=0.2)
    p.add_argument("--mi-samples", type=int, default=20000)
    p.add_argument("--mi-clip", type=float, default=5.0)

    # wandb
    p.add_argument("--wandb-project", type=str, default="")
    p.add_argument("--disable-wandb", action="store_true")

    return p.parse_args()


def create_experiment_dir(out_root: Path) -> Path:
    ts = datetime.now().strftime("run_%y%m%d_%H%M%S")
    exp_dir = out_root / ts
    (exp_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (exp_dir / "logs").mkdir(parents=True, exist_ok=True)
    return exp_dir


def save_config(exp_dir: Path, args: argparse.Namespace, device: torch.device, cfg_name: str):
    cfg_path = exp_dir / "config.txt"
    with cfg_path.open("w") as f:
        f.write(f"Created: {datetime.now().isoformat()}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Python: {sys.version}\n")
        f.write(f"PyTorch: {torch.__version__}\n")
        f.write(f"Config builder: {cfg_name}\n")
        f.write("\nArgs:\n")
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")


def maybe_init_wandb(args, input_shape, train_csv, val_csv, device):
    if args.disable_wandb:
        return None

    loss_to_project = {
        "mse": "tm-lesion-mse",
        "ncc": "tm-ncc-lesion",
        "mi":  "tm-lesion-mi",
    }
    wandb_project = args.wandb_project or loss_to_project.get(args.image_loss, "tm-unknown")

    run = wandb.init(
        project=wandb_project,
        config={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "lr": args.lr,
            "image_loss": args.image_loss,
            "sim_weight": args.sim_weight,
            "reg_weight": args.reg_weight,
            "axial_weight": args.axial_weight,
            "axial_x": args.axial_x,
            "axial_y": args.axial_y,
            "axial_z": args.axial_z,
            "axial_penalty": args.axial_penalty,
            "input_shape": input_shape,
            "window_size": tuple(args.window_size),
            "variant": args.transmorph_variant,
            "device": str(device),
            "train_csv": str(train_csv),
            "val_csv": str(val_csv),
            "exp_id": args.exp_id,
            "mi_bins": args.mi_bins,
            "mi_sigma": args.mi_sigma,
            "mi_samples": args.mi_samples,
            "mi_clip": args.mi_clip,
        },
    )

    exp_prefix = f"{args.exp_id}_" if args.exp_id else ""
    run.name = (
        exp_prefix
        + "tm"
        + f"_{args.transmorph_variant}"
        + f"_{args.image_loss}"
        + f"_reg{args.reg_weight:g}"
        + f"_ax{args.axial_weight:g}"
        + f"_lr{args.lr:g}"
        + f"_ep{args.epochs}"
    )

    if args.image_loss == "mi":
        run.name += f"_bins{args.mi_bins}_sig{args.mi_sigma:g}_samp{args.mi_samples}"

    return run


# ---------------- Training ---------------- #

def main():
    args = parse_args()

    root = Path(__file__).resolve().parents[1]
    train_csv = Path(args.train_csv).resolve()
    val_csv = Path(args.val_csv).resolve()
    out_root = Path(args.out_root).resolve()
    repo_root = Path(args.transmorph_repo).resolve()

    out_root.mkdir(parents=True, exist_ok=True)

    print("==== TransMorph Breast Biopsy Training ====")
    print(f"Script root     : {root}")
    print(f"TransMorph repo : {repo_root}")
    print(f"Train CSV       : {train_csv}")
    print(f"Val CSV         : {val_csv}")
    print(f"Out root        : {out_root}")
    print("-------------------------------------------")

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Device: {device}")
    if use_cuda:
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: CUDA not available, training will run on CPU.")

    torch.manual_seed(42)
    np.random.seed(42)
    if use_cuda:
        torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = True

    input_shape = tuple(args.input_shape)
    window_size = tuple(args.window_size)

    # import model
    TransMorphModel, tm_configs = import_transmorph(repo_root)
    cfg, cfg_name = get_config_by_variant(tm_configs, args.transmorph_variant)

    # override config for your data
    if hasattr(cfg, "img_size"):
        cfg.img_size = input_shape
    if hasattr(cfg, "window_size"):
        cfg.window_size = window_size
    if hasattr(cfg, "in_chans"):
        cfg.in_chans = 2

    print(f"Using config builder : {cfg_name}")
    print(f"Configured img_size  : {getattr(cfg, 'img_size', 'N/A')}")
    print(f"Configured window    : {getattr(cfg, 'window_size', 'N/A')}")

    model = TransMorphModel(cfg).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    run = maybe_init_wandb(args, input_shape, train_csv, val_csv, device)

    # datasets
    train_ds = BreastPairDataset(train_csv, target_shape=input_shape)
    val_ds = BreastPairDataset(val_csv, target_shape=input_shape)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
    )

    # losses
    mse_loss_fn = nn.MSELoss()

    if args.image_loss == "ncc":
        base_sim_loss_fn = GlobalNCCLoss()
        print("Using Global NCC loss (with MSE fallback on NaN/Inf).")
    elif args.image_loss == "mi":
        base_sim_loss_fn = SoftMILoss(
            bins=args.mi_bins,
            sigma=args.mi_sigma,
            samples=args.mi_samples,
            clip=args.mi_clip,
        )
        print("Using Soft MI loss (with MSE fallback on NaN/Inf).")
    else:
        base_sim_loss_fn = mse_loss_fn
        print("Using MSE loss.")

    reg_loss_fn = Grad3dLoss(penalty="l2")
    axial_loss_fn = AxialFlowLoss(
        wx=args.axial_x,
        wy=args.axial_y,
        wz=args.axial_z,
        penalty=args.axial_penalty,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(
        f"Axial loss config   : weight={args.axial_weight:g}, "
        f"axis_weights=({args.axial_x:g}, {args.axial_y:g}, {args.axial_z:g}), "
        f"penalty={args.axial_penalty}"
    )

    use_autocast = use_cuda and (args.image_loss == "mse")
    scaler = torch.cuda.amp.GradScaler(enabled=use_autocast)

    exp_dir = create_experiment_dir(out_root)
    ckpt_dir = exp_dir / "checkpoints"
    log_dir = exp_dir / "logs"

    save_config(exp_dir, args, device, cfg_name)
    writer = SummaryWriter(log_dir=str(log_dir))

    if run is not None:
        wandb.run.summary["num_train_pairs"] = len(train_ds)
        wandb.run.summary["num_val_pairs"] = len(val_ds)
        wandb.run.summary["exp_dir"] = str(exp_dir)
        wandb.run.summary["model_parameters"] = int(num_params)

    best_val_loss = float("inf")

    print(f"Experiment dir: {exp_dir}")
    print("Start training...\n")

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # ----- TRAIN ----- #
        model.train()
        train_loss_sum = 0.0
        train_sim_sum = 0.0
        train_reg_sum = 0.0
        train_axial_sum = 0.0
        train_batches = 0

        for mv, fx in train_loader:
            mv = mv.to(device, non_blocking=True)
            fx = fx.to(device, non_blocking=True)

            x_in = torch.cat((mv, fx), dim=1)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_autocast):
                warped, flow = model(x_in)

                sim_raw = base_sim_loss_fn(fx, warped)
                if not torch.isfinite(sim_raw):
                    print("[WARN] sim_loss (train) NaN/Inf -> fallback to MSE for this batch.")
                    sim_raw = mse_loss_fn(fx, warped)

                sim_loss = sim_raw * args.sim_weight
                reg_loss = reg_loss_fn(flow) * args.reg_weight
                axial_loss = axial_loss_fn(flow) * args.axial_weight
                loss = sim_loss + reg_loss + axial_loss

            if not torch.isfinite(loss):
                print("\n[ERROR] Non-finite loss detected in TRAIN phase.")
                print(f"  sim_loss: {sim_loss.detach().cpu().item()}")
                print(f"  reg_loss: {reg_loss.detach().cpu().item()}")
                print(f"  axial_loss: {axial_loss.detach().cpu().item()}")
                raise RuntimeError("Loss became NaN/Inf during training.")

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += float(loss.item())
            train_sim_sum += float(sim_loss.item())
            train_reg_sum += float(reg_loss.item())
            train_axial_sum += float(axial_loss.item())
            train_batches += 1

        avg_train_loss = train_loss_sum / max(train_batches, 1)
        avg_train_sim = train_sim_sum / max(train_batches, 1)
        avg_train_reg = train_reg_sum / max(train_batches, 1)
        avg_train_axial = train_axial_sum / max(train_batches, 1)

        # ----- VALIDATION ----- #
        model.eval()
        val_loss_sum = 0.0
        val_sim_sum = 0.0
        val_reg_sum = 0.0
        val_axial_sum = 0.0
        val_batches = 0

        with torch.no_grad():
            for mv, fx in val_loader:
                mv = mv.to(device, non_blocking=True)
                fx = fx.to(device, non_blocking=True)

                x_in = torch.cat((mv, fx), dim=1)

                with torch.cuda.amp.autocast(enabled=use_autocast):
                    warped, flow = model(x_in)

                    sim_raw = base_sim_loss_fn(fx, warped)
                    if not torch.isfinite(sim_raw):
                        print("[WARN] sim_loss (val) NaN/Inf -> fallback to MSE for this batch.")
                        sim_raw = mse_loss_fn(fx, warped)

                    sim_loss = sim_raw * args.sim_weight
                    reg_loss = reg_loss_fn(flow) * args.reg_weight
                    axial_loss = axial_loss_fn(flow) * args.axial_weight
                    loss = sim_loss + reg_loss + axial_loss

                if not torch.isfinite(loss):
                    print("\n[ERROR] Non-finite loss detected in VAL phase.")
                    print(f"  sim_loss: {sim_loss.detach().cpu().item()}")
                    print(f"  reg_loss: {reg_loss.detach().cpu().item()}")
                    print(f"  axial_loss: {axial_loss.detach().cpu().item()}")
                    raise RuntimeError("Loss became NaN/Inf during validation.")

                val_loss_sum += float(loss.item())
                val_sim_sum += float(sim_loss.item())
                val_reg_sum += float(reg_loss.item())
                val_axial_sum += float(axial_loss.item())
                val_batches += 1

        avg_val_loss = val_loss_sum / max(val_batches, 1)
        avg_val_sim = val_sim_sum / max(val_batches, 1)
        avg_val_reg = val_reg_sum / max(val_batches, 1)
        avg_val_axial = val_axial_sum / max(val_batches, 1)
        elapsed = time.time() - epoch_start

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} "
            f"- train_loss: {avg_train_loss:.4f} (sim {avg_train_sim:.4f}, reg {avg_train_reg:.4f}, axial {avg_train_axial:.4f}) "
            f"- val_loss: {avg_val_loss:.4f} (sim {avg_val_sim:.4f}, reg {avg_val_reg:.4f}, axial {avg_val_axial:.4f}) "
            f"- time: {elapsed/60:.1f} min"
        )

        writer.add_scalar("Loss/train_total", avg_train_loss, epoch)
        writer.add_scalar("Loss/train_sim", avg_train_sim, epoch)
        writer.add_scalar("Loss/train_reg", avg_train_reg, epoch)
        writer.add_scalar("Loss/train_axial", avg_train_axial, epoch)
        writer.add_scalar("Loss/val_total", avg_val_loss, epoch)
        writer.add_scalar("Loss/val_sim", avg_val_sim, epoch)
        writer.add_scalar("Loss/val_reg", avg_val_reg, epoch)
        writer.add_scalar("Loss/val_axial", avg_val_axial, epoch)

        if run is not None:
            wandb.log(
                {
                    "epoch": epoch,
                    "loss/train_total": avg_train_loss,
                    "loss/train_sim": avg_train_sim,
                    "loss/train_reg": avg_train_reg,
                    "loss/train_axial": avg_train_axial,
                    "loss/val_total": avg_val_loss,
                    "loss/val_sim": avg_val_sim,
                    "loss/val_reg": avg_val_reg,
                    "loss/val_axial": avg_val_axial,
                    "time/epoch_min": elapsed / 60.0,
                },
                step=epoch,
            )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = ckpt_dir / "best_model.pth"
            torch.save(model.state_dict(), best_path)
            print(f"  -> New best model saved to {best_path} (val_loss={best_val_loss:.4f})")

            if run is not None:
                wandb.run.summary["best_val_loss"] = best_val_loss
                wandb.run.summary["best_model_path"] = str(best_path)

        if (epoch % args.save_every == 0) or (epoch == args.epochs):
            ckpt_path = ckpt_dir / f"checkpoint_epoch_{epoch:03d}.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss": avg_val_loss,
                    "variant": args.transmorph_variant,
                    "input_shape": input_shape,
                    "window_size": window_size,
                },
                ckpt_path,
            )
            print(f"  -> Checkpoint saved to {ckpt_path}")

    writer.close()
    if run is not None:
        wandb.finish()

    print("\nTraining finished.")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Experiment directory: {exp_dir}")


if __name__ == "__main__":
    main()