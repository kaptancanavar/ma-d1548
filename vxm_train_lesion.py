#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Training script for 3D VoxelMorph (PyTorch) on preprocessed MR breast biopsy data.

- Uses train_pairs.csv / val_pairs.csv with columns:
  split,patient_id,study_id,breast_side,moving,fixed,biopsy_img,marker_img

Assumes:
  * 'moving' and 'fixed' paths point to preprocessed NIfTI volumes
  * volumes have shape (224, 224, 96) by default
  * data are already RAS + z-score normalised + cropped + resized

Outputs:
  --out-root/run_<timestamp>/
    config.txt
    logs/
    checkpoints/
      checkpoint_epoch_XXX.pth
      best_model.pth

W&B project names:
  mse -> vxm-lesion-mse
  ncc -> vxm-ncc-lesion
  mi  -> vxm-lesion-mi
"""

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

# Make sure VoxelMorph uses PyTorch backend
os.environ["NEURITE_BACKEND"] = "pytorch"
os.environ["VXM_BACKEND"] = "pytorch"
import voxelmorph as vxm  # type: ignore


# ---------------- Dataset ---------------- #

class BreastPairDataset(Dataset):
    """
    Dataset that reads moving/fixed pairs from a CSV file.
    CSV must contain at least: moving,fixed
    """

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

        # robustify: remove NaN/Inf & clamp
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        # for z-score-normalised volumes, values should roughly be in [-5, 5]
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
    Returns loss = -NCC (minimize loss -> maximize NCC).
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
    Differentiable (soft) Mutual Information loss via soft histograms.
    Uses voxel subsampling for speed.

    Returns loss = -MI (minimize loss -> maximize MI).

    Notes:
    - Assumes intensities roughly within [-clip, clip] (for z-scored data: clip~5).
    - If too slow / OOM: reduce --mi-samples or --mi-bins.
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

        # soft assignment
        sigma = max(self.sigma, 1e-4)
        xw = torch.exp(-0.5 * ((x.unsqueeze(-1) - centers) / sigma) ** 2)
        yw = torch.exp(-0.5 * ((y.unsqueeze(-1) - centers) / sigma) ** 2)

        xw = xw / (xw.sum(dim=-1, keepdim=True) + self.eps)
        yw = yw / (yw.sum(dim=-1, keepdim=True) + self.eps)

        # joint histogram p(x,y): (B,bins,bins)
        pxy = torch.bmm(xw.transpose(1, 2), yw) / xw.shape[1]
        px = pxy.sum(dim=2, keepdim=True)
        py = pxy.sum(dim=1, keepdim=True)

        mi = pxy * (torch.log(pxy + self.eps) - torch.log(px * py + self.eps))
        mi = mi.sum(dim=(1, 2))
        return -mi.mean()


# ---------------- Utility ---------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train 3D VoxelMorph on breast biopsy MRIs.")

    # data & I/O
    p.add_argument("--train-csv", type=str, default="data/train_pairs.csv")
    p.add_argument("--val-csv", type=str, default="data/val_pairs.csv")
    p.add_argument("--out-root", type=str, default="experiments")

    # training hyperparameters
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)

    p.add_argument(
        "--image-loss",
        type=str,
        default="mse",
        choices=["ncc", "mse", "mi"],
        help="Image similarity loss.",
    )
    p.add_argument("--sim-weight", type=float, default=1.0)
    p.add_argument("--reg-weight", type=float, default=1.0)
    p.add_argument("--grad-downsample", type=int, default=1, help="loss_mult for Grad loss (CVPR baseline=1).")

    p.add_argument("--input-shape", type=int, nargs=3, default=[224, 224, 96], metavar=("NX", "NY", "NZ"))
    p.add_argument("--save-every", type=int, default=10)

    p.add_argument("--bidir", action="store_true")
    p.add_argument("--int-steps", type=int, default=0)

    p.add_argument("--exp-id", type=str, default="")

    # MI hyperparameters (used only if --image-loss mi)
    p.add_argument("--mi-bins", type=int, default=32)
    p.add_argument("--mi-sigma", type=float, default=0.2)
    p.add_argument("--mi-samples", type=int, default=20000)
    p.add_argument("--mi-clip", type=float, default=5.0)

    # W&B optional override
    p.add_argument("--wandb-project", type=str, default="", help="Override W&B project name (optional).")

    return p.parse_args()


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
        f.write(f"VoxelMorph: {vxm.__version__ if hasattr(vxm, '__version__') else 'unknown'}\n")
        f.write("\nArgs:\n")
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")


# ---------------- Training ---------------- #

def main():
    args = parse_args()

    root = Path(__file__).resolve().parents[1]
    train_csv = (root / args.train_csv).resolve() if not Path(args.train_csv).is_absolute() else Path(args.train_csv).resolve()
    val_csv = (root / args.val_csv).resolve() if not Path(args.val_csv).is_absolute() else Path(args.val_csv).resolve()
    out_root = (root / args.out_root).resolve() if not Path(args.out_root).is_absolute() else Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    print("==== VoxelMorph Breast Biopsy Training ====")
    print(f"Root dir : {root}")
    print(f"Train CSV: {train_csv}")
    print(f"Val CSV  : {val_csv}")
    print(f"Out root : {out_root}")
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

    # ---- W&B project per loss (exact names you requested) ----
    loss_to_project = {
        "mse": "vxm-lesion-mse",
        "ncc": "vxm-ncc-lesion",
        "mi":  "vxm-lesion-mi",
    }
    wandb_project = args.wandb_project or loss_to_project.get(args.image_loss, "vxm-unknown")
    print(f"[W&B] Using project: {wandb_project}")

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
            "grad_downsample": args.grad_downsample,
            "input_shape": input_shape,
            "bidir": args.bidir,
            "int_steps": args.int_steps,
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
        + "vxm"
        + f"_{args.image_loss}"
        + f"_reg{args.reg_weight:g}"
        + f"_lr{args.lr:g}"
        + f"_int{args.int_steps}"
        + f"_bidir{int(args.bidir)}"
        + f"_ep{args.epochs}"
    )
    if args.image_loss == "mi":
        run.name += f"_bins{args.mi_bins}_sig{args.mi_sigma:g}_samp{args.mi_samples}"

    # datasets & loaders
    train_ds = BreastPairDataset(train_csv, target_shape=input_shape)
    val_ds = BreastPairDataset(val_csv, target_shape=input_shape)

    wandb.run.summary["num_train_pairs"] = len(train_ds)
    wandb.run.summary["num_val_pairs"] = len(val_ds)

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

    # model
    enc_nf = [16, 32, 32, 32]
    dec_nf = [32, 32, 32, 32, 32, 16]
    nb_features = [enc_nf, dec_nf]

    model = vxm.networks.VxmDense(
        inshape=input_shape,
        nb_unet_features=nb_features,
        int_steps=args.int_steps,
        int_downsize=2,
        bidir=args.bidir,
    ).to(device)

    # losses (with MSE fallback)
    mse_loss_fn = vxm.losses.MSE().loss

    if args.image_loss == "ncc":
        base_sim_loss_fn = GlobalNCCLoss()
        print("Using Global NCC loss (with MSE fallback on NaN/Inf).")
    elif args.image_loss == "mi":
        base_sim_loss_fn = SoftMILoss(
            bins=args.mi_bins, sigma=args.mi_sigma, samples=args.mi_samples, clip=args.mi_clip
        )
        print("Using Soft MI loss (with MSE fallback on NaN/Inf).")
    else:
        base_sim_loss_fn = mse_loss_fn
        print("Using MSE loss.")

    reg_loss_fn = vxm.losses.Grad("l2", loss_mult=args.grad_downsample).loss

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # AMP only for MSE (NCC/MI are more fragile in mixed precision)
    if use_cuda:
        scaler = torch.amp.GradScaler("cuda")
        autocast_device = "cuda"
    else:
        scaler = torch.amp.GradScaler("cpu")
        autocast_device = "cpu"
    use_autocast = use_cuda and (args.image_loss == "mse")

    # experiment dirs & logging
    exp_dir = create_experiment_dir(out_root)
    ckpt_dir = exp_dir / "checkpoints"
    log_dir = exp_dir / "logs"

    save_config(exp_dir, args, device)
    writer = SummaryWriter(log_dir=str(log_dir))
    wandb.run.summary["exp_dir"] = str(exp_dir)

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
        train_batches = 0

        for mv, fx in train_loader:
            mv = mv.to(device)
            fx = fx.to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(autocast_device, enabled=use_autocast):
                out = model(mv, fx)
                if args.bidir:
                    warp_m2f, warp_f2m, flow_m2f, flow_f2m = out
                    sim_raw = base_sim_loss_fn(fx, warp_m2f)
                    if not torch.isfinite(sim_raw):
                        print("[WARN] sim_loss (train) NaN/Inf -> fallback to MSE for this batch.")
                        sim_raw = mse_loss_fn(fx, warp_m2f)
                    sim_loss = sim_raw * args.sim_weight
                    reg_loss = reg_loss_fn(None, flow_m2f) * args.reg_weight
                else:
                    warp, flow = out
                    sim_raw = base_sim_loss_fn(fx, warp)
                    if not torch.isfinite(sim_raw):
                        print("[WARN] sim_loss (train) NaN/Inf -> fallback to MSE for this batch.")
                        sim_raw = mse_loss_fn(fx, warp)
                    sim_loss = sim_raw * args.sim_weight
                    reg_loss = reg_loss_fn(None, flow) * args.reg_weight

                loss = sim_loss + reg_loss

            if not torch.isfinite(loss):
                print("\n[ERROR] Non-finite loss detected in TRAIN phase (even after fallback).")
                print(f"  sim_loss: {sim_loss.detach().cpu().numpy()}")
                print(f"  reg_loss: {reg_loss.detach().cpu().numpy()}")
                raise RuntimeError("Loss became NaN/Inf during training.")

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += float(loss.item())
            train_sim_sum += float(sim_loss.item())
            train_reg_sum += float(reg_loss.item())
            train_batches += 1

        avg_train_loss = train_loss_sum / max(train_batches, 1)
        avg_train_sim = train_sim_sum / max(train_batches, 1)
        avg_train_reg = train_reg_sum / max(train_batches, 1)

        # ----- VALIDATION ----- #
        model.eval()
        val_loss_sum = 0.0
        val_sim_sum = 0.0
        val_reg_sum = 0.0
        val_batches = 0

        with torch.no_grad():
            for mv, fx in val_loader:
                mv = mv.to(device)
                fx = fx.to(device)

                with torch.amp.autocast(autocast_device, enabled=use_autocast):
                    out = model(mv, fx)
                    if args.bidir:
                        warp_m2f, warp_f2m, flow_m2f, flow_f2m = out
                        sim_raw = base_sim_loss_fn(fx, warp_m2f)
                        if not torch.isfinite(sim_raw):
                            print("[WARN] sim_loss (val) NaN/Inf -> fallback to MSE for this batch.")
                            sim_raw = mse_loss_fn(fx, warp_m2f)
                        sim_loss = sim_raw * args.sim_weight
                        reg_loss = reg_loss_fn(None, flow_m2f) * args.reg_weight
                    else:
                        warp, flow = out
                        sim_raw = base_sim_loss_fn(fx, warp)
                        if not torch.isfinite(sim_raw):
                            print("[WARN] sim_loss (val) NaN/Inf -> fallback to MSE for this batch.")
                            sim_raw = mse_loss_fn(fx, warp)
                        sim_loss = sim_raw * args.sim_weight
                        reg_loss = reg_loss_fn(None, flow) * args.reg_weight

                    loss = sim_loss + reg_loss

                if not torch.isfinite(loss):
                    print("\n[ERROR] Non-finite loss detected in VAL phase (even after fallback).")
                    print(f"  sim_loss: {sim_loss.detach().cpu().numpy()}")
                    print(f"  reg_loss: {reg_loss.detach().cpu().numpy()}")
                    raise RuntimeError("Loss became NaN/Inf during validation.")

                val_loss_sum += float(loss.item())
                val_sim_sum += float(sim_loss.item())
                val_reg_sum += float(reg_loss.item())
                val_batches += 1

        avg_val_loss = val_loss_sum / max(val_batches, 1)
        avg_val_sim = val_sim_sum / max(val_batches, 1)
        avg_val_reg = val_reg_sum / max(val_batches, 1)
        elapsed = time.time() - epoch_start

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} "
            f"- train_loss: {avg_train_loss:.4f} (sim {avg_train_sim:.4f}, reg {avg_train_reg:.4f}) "
            f"- val_loss: {avg_val_loss:.4f} (sim {avg_val_sim:.4f}, reg {avg_val_reg:.4f}) "
            f"- time: {elapsed/60:.1f} min"
        )

        writer.add_scalar("Loss/train_total", avg_train_loss, epoch)
        writer.add_scalar("Loss/train_sim", avg_train_sim, epoch)
        writer.add_scalar("Loss/train_reg", avg_train_reg, epoch)
        writer.add_scalar("Loss/val_total", avg_val_loss, epoch)
        writer.add_scalar("Loss/val_sim", avg_val_sim, epoch)
        writer.add_scalar("Loss/val_reg", avg_val_reg, epoch)

        wandb.log(
            {
                "epoch": epoch,
                "loss/train_total": avg_train_loss,
                "loss/train_sim": avg_train_sim,
                "loss/train_reg": avg_train_reg,
                "loss/val_total": avg_val_loss,
                "loss/val_sim": avg_val_sim,
                "loss/val_reg": avg_val_reg,
                "time/epoch_min": elapsed / 60.0,
            },
            step=epoch,
        )

        # save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = ckpt_dir / "best_model.pth"
            torch.save(model.state_dict(), best_path)
            print(f"  -> New best model saved to {best_path} (val_loss={best_val_loss:.4f})")
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
                },
                ckpt_path,
            )
            print(f"  -> Checkpoint saved to {ckpt_path}")

    writer.close()
    wandb.finish()

    print("\nTraining finished.")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Experiment directory: {exp_dir}")


if __name__ == "__main__":
    main()
