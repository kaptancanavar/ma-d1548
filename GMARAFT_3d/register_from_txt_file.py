# eval_register_stack_with_network_matched_to_training.py

__author__ = "Semih Tarik Uenal"

import os
import json
import numpy as np
import torch
import torch.nn.functional as F

from io import StringIO
from contextlib import redirect_stdout

from torch.utils.data import DataLoader  # (not used, kept for parity)
from preprocessing import MetaImageIO
from network_3d.model import GMARAFT_Denoiser3D
import  bspline_interp_withgrad
# --------- OPTIONAL: B-spline inversion helpers ----------
import bspline_interp

def pad_image_for_bspline(img: torch.Tensor, padding: int = 2) -> torch.Tensor:
    """
    Pad (Z, Y, X) axes symmetrically for cubic B-spline support.
    """
    return torch.nn.functional.pad(img, pad=(padding, padding, padding, padding, padding, padding), mode='reflect')



def warp_3d_batch_bspline(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """
    Warp a 3D real-valued image using B-spline interpolation and voxel-space motion fields.

    Args:
        img (torch.Tensor): Complex input image of shape (M, D, H, W), where:
            - M: number of motion states
            - D: depth (Lin)
            - H: height (Par)
            - W: width (Col)
        flow (torch.Tensor): Voxel-space displacement fields of shape (M, D, H, W, 3),
            where the last dimension corresponds to (ΔZ, ΔY, ΔX) displacements.

    Returns:
        torch.Tensor: Warped complex image of shape (M, D, H, W)
    """
    pad = 2
    if img.shape[0] ==1:
        dim_to_squeeze = 0
    else:
        dim_to_squeeze = 1
    
    img = img.squeeze(dim_to_squeeze)
    flow = flow.permute(0,2,3,4,1).contiguous()  # (M,D,H,W,3) in (dx,dy,dz)                
    M, D, H, W = img.shape


    img_real = pad_image_for_bspline(img, pad)
    img_real = bspline_interp_withgrad.bspline_prefilter_autograd(
        img_real.contiguous()
    )

    z = torch.arange(D, device=img.device)
    y = torch.arange(H, device=img.device)
    x = torch.arange(W, device=img.device)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')
    grid = torch.stack((zz, yy, xx), dim=-1)  # (D,H,W,3)
    grid = grid.unsqueeze(0).repeat(M,1,1,1,1).float()

    coords = grid + flow + pad

    warped_real = bspline_interp_withgrad.bspline_interp_autograd(
        img_real.contiguous(),
        coords.contiguous()
    )

    return warped_real.unsqueeze(dim_to_squeeze)


def pad_image_for_bspline(img: torch.Tensor, padding: int = 2) -> torch.Tensor:
    # img: (D,H,W)
    return F.pad(img, pad=(padding, padding, padding, padding, padding, padding), mode="reflect")


def warp_displacement_bspline(fwd_field: torch.Tensor, inv_guess: torch.Tensor, pad: int = 2) -> torch.Tensor:
    """
    Warp forward displacement field u at points (x + v) using cubic B-spline interpolation.

    fwd_field: (M, D, H, W, 3) voxel displacement
    inv_guess: (M, D, H, W, 3) voxel displacement (sampling offset)
    return:    (M, D, H, W, 3)
    """
    assert fwd_field.ndim == 5 and inv_guess.ndim == 5
    M, D, H, W, _ = fwd_field.shape

    z = torch.arange(D, device=fwd_field.device)
    y = torch.arange(H, device=fwd_field.device)
    x = torch.arange(W, device=fwd_field.device)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    coords = torch.stack((zz, yy, xx), dim=-1).float().unsqueeze(0).repeat(M, 1, 1, 1, 1)  # (M,D,H,W,3)
    coords = coords + inv_guess + pad  # shift because of padding

    fwd_padded = torch.zeros((M, D + 2 * pad, H + 2 * pad, W + 2 * pad, 3),
                             dtype=fwd_field.dtype, device=fwd_field.device)
    for c in range(3):
        fwd_padded[..., c] = pad_image_for_bspline(fwd_field[..., c], pad)
        bspline_interp.bspline_prefilter(fwd_padded[..., c].contiguous())

    warped = torch.zeros_like(fwd_field)
    for c in range(3):
        tmp = torch.zeros_like(fwd_field[..., 0])
        bspline_interp.bspline_interp(fwd_padded[..., c].contiguous(), coords.contiguous(), tmp)
        warped[..., c] = tmp
    return warped


def invert_displacement_fixed_point_3d(fwd_field: torch.Tensor, num_iters: int = 40) -> torch.Tensor:
    """
    Fixed-point inversion v = -u(x + v).
    fwd_field: (M, D, H, W, 3)
    return:    (M, D, H, W, 3)
    """
    v = torch.zeros_like(fwd_field)
    for _ in range(num_iters):
        u_xpv = warp_displacement_bspline(fwd_field, v)
        v = -u_xpv
    return v


@torch.no_grad()
def validate_inverse_field(fwd_field: torch.Tensor, inv_field: torch.Tensor, pad: int = 2):
    """
    Prints mean and max of residual r = u(x+v) + v (should be ~0).
    """
    u_at_x_plus_v = warp_displacement_bspline(fwd_field, inv_field, pad=pad)
    residual = u_at_x_plus_v + inv_field
    err = torch.norm(residual, dim=-1)
    print("Inverse validation: mean=", err.mean().item(), " max=", err.max().item())
    return err
# ---------------------------------------------------------


# ----------------- helpers to MATCH DataLoader -----------------
def normalize(vol_t: torch.Tensor) -> torch.Tensor:
    """
    vol_t: (T,D,H,W) or (1,D,H,W) or (D,H,W)
    Per-frame min-max to [0,1].
    """
    if vol_t.ndim == 3:
        vol_t = vol_t.unsqueeze(0)
    assert vol_t.ndim == 4
    vmin = vol_t.amin(dim=(1, 2, 3), keepdim=True)
    vmax = vol_t.amax(dim=(1, 2, 3), keepdim=True)
    return (vol_t - vmin) / (vmax - vmin + 1e-8)


def _normalize01_single(vol_3d: torch.Tensor) -> torch.Tensor:
    # (D,H,W) -> [0,1]
    vmin = vol_3d.min()
    vmax = vol_3d.max()
    return (vol_3d - vmin) / (vmax - vmin + 1e-8)


def _apply_margin(bbox, shape, margin):
    # shape: (D,H,W); margin: (mz, mx, my)  <== matches your DataLoader naming/order
    (D, H, W) = shape
    (mz, mx, my) = margin
    (z0, z1), (x0, x1), (y0, y1) = bbox
    z0, z1 = max(0, z0 - mz), min(D, z1 + mz)
    x0, x1 = max(0, x0 - mx), min(H, x1 + mx)
    y0, y1 = max(0, y0 - my), min(W, y1 + my)
    return [[z0, z1], [x0, x1], [y0, y1]]


def bbox_from_volume_union_like_loader(stack_norm: torch.Tensor,
                                       k_mad: float = 15.0,
                                       min_foreground_q: float = 0.20,
                                       morph_iter: int = 2,
                                       crop_margin=(8, 16, 16)):
    """
    Replicates VibeDatasetPairwise3D._bbox_from_volume_union on normalized stack.

    stack_norm: (T,D,H,W) already per-frame normalized to [0,1]
    returns bbox as [[z0,z1],[x0,x1],[y0,y1]]
    """
    assert stack_norm.ndim == 4
    # Union max over time
    v = stack_norm.amax(dim=0)  # (D,H,W)
    v = _normalize01_single(v)
    v = torch.clamp(v, max=torch.quantile(v, 0.995))  # outlier suppression

    med = torch.median(v)
    mad = torch.median((v - med).abs()) + 1e-6
    t1 = med + k_mad * mad
    t2 = torch.quantile(v, min_foreground_q)
    thr = torch.max(t1, t2)

    mask = (v > thr).float()[None, None]  # (1,1,D,H,W)
    # morphological close (dilate then erode) with 3x3x3 max-pools
    for _ in range(morph_iter):
        mask = F.max_pool3d(mask, kernel_size=3, stride=1, padding=1)
    for _ in range(morph_iter):
        mask = 1.0 - F.max_pool3d(1.0 - mask, kernel_size=3, stride=1, padding=1)

    m = (mask[0, 0] > 0.5)
    if not m.any():
        D, H, W = v.shape
        bbox = [[0, D], [0, H], [0, W]]
    else:
        z, x, y = torch.where(m)
        bbox = [[int(z.min()), int(z.max()) + 1],
                [int(x.min()), int(x.max()) + 1],
                [int(y.min()), int(y.max()) + 1]]

    bbox = _apply_margin(bbox, v.shape, crop_margin)
    return bbox


def crop_zhy(vol: torch.Tensor, bbox):
    """
    vol: (1,D,H,W) -> cropped (1,d,h,w)
    bbox: [[z0,z1],[x0,x1],[y0,y1]]
    """
    (z0, z1), (x0, x1), (y0, y1) = bbox
    return vol[:, z0:z1, x0:x1, y0:y1]


def paste_flow(full_shape, flow_crop: torch.Tensor, bbox):
    """
    full_shape: (D,H,W)
    flow_crop: (1,3,d,h,w)
    returns: (1,3,D,H,W) with zeros outside bbox
    """
    full = torch.zeros((1, 3, *full_shape), device=flow_crop.device, dtype=flow_crop.dtype)
    (z0, z1), (x0, x1), (y0, y1) = bbox
    full[..., z0:z1, x0:x1, y0:y1] = flow_crop
    return full


def resize_volume_3d(x: torch.Tensor, size, align_corners=True):
    """
    x: (1,D,H,W) or (B,1,D,H,W). Returns same rank with new spatial size.
    """
    need_squeeze = (x.ndim == 4)
    if need_squeeze:
        x = x.unsqueeze(0)  # -> (B=1,1,D,H,W)
    out = F.interpolate(x, size=size, mode='trilinear', align_corners=align_corners)
    return out.squeeze(0) if need_squeeze else out


# def resize_flow_3d(flow: torch.Tensor, size_out, align_corners=True):
#     """
#     flow: (1,3,D_in,H_in,W_in) in voxel units of input grid
#     return: (1,3,D_out,H_out,W_out) with vector components scaled appropriately.
#     """
#     B, C, D_in, H_in, W_in = flow.shape
#     assert B == 1 and C == 3
#     Dz, Dh, Dw = size_out

#     flow_rs = F.interpolate(flow, size=(Dz, Dh, Dw), mode='trilinear', align_corners=align_corners)
#     sz = Dz / max(D_in, 1)
#     sy = Dh / max(H_in, 1)
#     sx = Dw / max(W_in, 1)
#     scale = torch.tensor([sz, sy, sx], device=flow.device, dtype=flow.dtype).view(1, 3, 1, 1, 1)
#     return flow_rs * scale
# -------------------------------------------------------------
def resize_flow_3d(flow: torch.Tensor, size_out, align_corners=True):
    B, C, D_in, H_in, W_in = flow.shape
    assert B == 1 and C == 3
    Dz, Dh, Dw = size_out
    flow_rs = F.interpolate(flow, size=(Dz, Dh, Dw), mode='trilinear', align_corners=align_corners)

    def sf(o, i):
        if o <= 1 or i <= 1:
            return 1.0
        return (o - 1) / (i - 1) if align_corners else o / i

    scale = torch.tensor([sf(Dz, D_in), sf(Dh, H_in), sf(Dw, W_in)],
                         device=flow.device, dtype=flow.dtype).view(1, 3, 1, 1, 1)
    return flow_rs * scale


@torch.no_grad()
def predict_flow_on_training_grid(model, ref_cropN: torch.Tensor, mov_cropN: torch.Tensor, target_shape=(32, 176, 176)):
    """
    ref_cropN, mov_cropN: (1,Dc,Hc,Wc) normalized, cropped
    1) resize to target_shape with align_corners=True (exactly as in training)
    2) run model
    returns: flow at target_shape (1,3,Dt,Ht,Wt)
    """
    ref_in = resize_volume_3d(ref_cropN, target_shape, align_corners=True).unsqueeze(0)  # (B=1,1,Dt,Ht,Wt)
    mov_in = resize_volume_3d(mov_cropN, target_shape, align_corners=True).unsqueeze(0)

    flow_low, flow_pr = model(ref_in, mov_in, test_mode=1)
    flow_s = flow_pr[-1] if isinstance(flow_pr, (list, tuple)) else flow_pr  # (1,3,Dt,Ht,Wt)
    return flow_s



def register_single_case(root_dir: str, config_path: str):
    """Run registration for a single case and capture stdout logs."""
    log_stream = StringIO()
    with redirect_stdout(log_stream):  # Capture all prints
        try:
            # ------------------ paths & config ------------------
            img_path = os.path.join(root_dir, "reconstructed_img_fista_direct_abs.mhd")

            # Load config
            with open(config_path, "r") as f:
                config = json.load(f)

            # Dataset params
            ds_cfg = config.get("dataset", config)
            target_shape = tuple(ds_cfg.get("target_shape", [32, 144, 144]))
            crop_margin = tuple(ds_cfg.get("crop_margin", [8, 16, 16]))
            morph_iter = int(ds_cfg.get("morph_iter", 2))
            min_foreground_q = float(ds_cfg.get("min_foreground_q", 0.20))
            k_mad = float(ds_cfg.get("k_mad", 15.0))

            checkpoint_path = os.path.join(
                os.getcwd(), config["trainer"]["save_dir"], config["name"], "70_final.pth"
            )

            # ------------------ load stack ------------------
            img_np = np.abs(MetaImageIO.read(img_path)).squeeze()
            assert img_np.ndim == 4, f"Expected (T,D,H,W), got {img_np.shape}"
            num_states, D, H, W = img_np.shape
            print(f"[info] stack loaded: states={num_states}, D={D}, H={H}, W={W}")

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            stack_t = torch.from_numpy(img_np).float().to(device)
            stack_norm = normalize(stack_t)

            bbox = bbox_from_volume_union_like_loader(
                stack_norm, k_mad=k_mad, min_foreground_q=min_foreground_q,
                morph_iter=morph_iter, crop_margin=crop_margin
            )
            (z0, z1), (x0, x1), (y0, y1) = bbox
            print(f"[info] bbox (z:{z0}-{z1}, x:{x0}-{x1}, y:{y0}-{y1})")

            fixed_full = stack_t[3].unsqueeze(0)
            fixed_crop = crop_zhy(fixed_full, bbox)
            fixed_cropN = normalize(fixed_crop)

            # ------------------ load network ------------------
            model = GMARAFT_Denoiser3D().to(device).eval()
            ckpt = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(ckpt["state_dict"])
            print(f"[info] model loaded: {checkpoint_path}")

            registered_volume = np.zeros_like(img_np, dtype=np.float32)
            deformation_fields_fwd = np.zeros((num_states, D, H, W, 3), np.float32)
            deformation_fields_inv = np.zeros((num_states, D, H, W, 3), np.float32)

            registered_volume[0] = img_np[0]
            deformation_fields_fwd[0] = 0.0
            deformation_fields_inv[0] = 0.0

            for i in range(num_states):
                print(f"[run] Registering motion state {i+1}/{num_states} ...")

                moving_full = stack_t[i].unsqueeze(0)
                moving_crop = crop_zhy(moving_full, bbox)
                moving_cropN = normalize(moving_crop)

                flow_target = predict_flow_on_training_grid(
                    model, fixed_cropN, moving_cropN, target_shape=target_shape
                )

                dc, hc, wc = moving_crop.shape[1:]
                flow_crop = resize_flow_3d(flow_target, (dc, hc, wc), align_corners=True)
                flow_full = paste_flow((D, H, W), flow_crop, bbox)

                warped = warp_3d_batch_bspline(moving_full.unsqueeze(0), flow_full[:, [2, 1, 0], ...])
                registered_volume[i] = warped.squeeze().detach().cpu().numpy()

                flow_vox_dhwd3 = flow_full[0].permute(1, 2, 3, 0).detach().cpu().numpy()
                deformation_fields_fwd[i] = flow_vox_dhwd3

                flow_for_inv = torch.from_numpy(flow_vox_dhwd3[None, ...]).to(device).float()
                inv_field = invert_displacement_fixed_point_3d(flow_for_inv, num_iters=40)
                validate_inverse_field(flow_for_inv, inv_field, pad=2)
                deformation_fields_inv[i] = inv_field.squeeze(0).detach().cpu().numpy()

                torch.cuda.empty_cache()

            reg_path = os.path.join(root_dir, "registered_network.mhd")
            fwd_path = os.path.join(root_dir, "deformation_fields_fwd_network.mhd")
            inv_path = os.path.join(root_dir, "deformation_fields_inv_network.mhd")

            MetaImageIO.write(reg_path, registered_volume)
            MetaImageIO.write(fwd_path, deformation_fields_fwd)
            MetaImageIO.write(inv_path, deformation_fields_inv)

            print("\n✅ Saved:")
            print("  → Registered Volume         :", reg_path)
            print("  → Forward Deformation Field :", fwd_path)
            print("  → Inverse Deformation Field :", inv_path)

            fwd_mag = np.linalg.norm(deformation_fields_fwd, axis=-1)
            inv_mag = np.linalg.norm(deformation_fields_inv, axis=-1)
            print("\n🔍 Forward field stats:")
            print(f"   shape={deformation_fields_fwd.shape}, "
                  f"min={fwd_mag.min():.4f}, max={fwd_mag.max():.4f}, mean={fwd_mag.mean():.4f}")
            print("🔍 Inverse field stats:")
            print(f"   shape={deformation_fields_inv.shape}, "
                  f"min={inv_mag.min():.4f}, max={inv_mag.max():.4f}, mean={inv_mag.mean():.4f}")

        except Exception as e:
            print(f"[ERROR] Failed to process {root_dir}: {e}")

    # Write log to file
    log_text = log_stream.getvalue()
    log_path = os.path.join(root_dir, "registration_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(log_text)

    print(f"[log] written: {log_path}")


def main_batch():
    cwd = os.getcwd()
    files_txt = os.path.join(cwd, "val_files.txt")
    config_path = os.path.join(cwd, "configs", "test_vibe_pairwise.json")

    if not os.path.exists(files_txt):
        print(f"[error] {files_txt} not found")
        return

    with open(files_txt, "r") as f:
        all_paths = [line.strip() for line in f if line.strip()]

    print(f"[info] Found {len(all_paths)} paths in ._files.txt")

    for path in all_paths:
        folder_name = os.path.basename(path).lower()
        if "bh" in folder_name:
            print(f"[skip] {path} (contains 'bh')")
            continue
        if not os.path.isdir(path):
            print(f"[warn] Skipping invalid path: {path}")
            continue

        print(f"[run] Processing: {path}")
        register_single_case(path, config_path)


if __name__ == "__main__":
    main_batch()


