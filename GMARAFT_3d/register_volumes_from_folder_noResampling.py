# eval_register_stack_with_network_matched_to_training.py

__author__ = "Semih Tarik Uenal"

import os
import json
import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader  # (not used, kept for parity)
from preprocessing import MetaImageIO
from network_3d.model import GMARAFT_Denoiser3D
from train.warp import warp_3d_batch_bspline, warp_3d_torch

# --------- OPTIONAL: B-spline inversion helpers ----------
import bspline_interp


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
# -------------------------------------------------------------


# ------------------ NEW: pad/crop utilities (no resampling) ------------------
def _compute_pad_crop(in_shape, target_shape):
    """
    Returns per-dim (z,x,y):
        pads   = ((pz0,pz1),(px0,px1),(py0,py1))  # reflect-pad sizes applied to input
        crops  = ((cz0,cz1),(cx0,cx1),(cy0,cy1))  # [start,end) indices taken from *original* input
    A dim may have either padding or cropping (or neither). Use together to invert.
    """
    (D, H, W) = in_shape
    (Td, Th, Tw) = target_shape

    pads = []
    crops = []

    for i_in, i_tar in zip((D, H, W), (Td, Th, Tw)):
        if i_in == i_tar:
            pads.append((0, 0))
            crops.append((0, i_in))  # identity
        elif i_in < i_tar:
            # center pad to reach target
            total = i_tar - i_in
            p0 = total // 2
            p1 = total - p0
            pads.append((p0, p1))
            crops.append((0, i_in))
        else:
            # center crop to target
            start = (i_in - i_tar) // 2
            end = start + i_tar
            pads.append((0, 0))
            crops.append((start, end))

    return tuple(pads), tuple(crops)


def pad_or_crop_volume(vol_1dhw: torch.Tensor, target_shape, mode="reflect"):
    """
    vol_1dhw: (1,D,H,W)
    Returns:
      vol_t: (1,Td,Th,Tw) after center reflect-pad and/or center-crop
      spec:  dict with keys 'pads','crops','in_shape','target_shape' for inversion
    """
    assert vol_1dhw.ndim == 4 and vol_1dhw.shape[0] == 1
    in_shape = vol_1dhw.shape[1:]
    pads, crops = _compute_pad_crop(in_shape, target_shape)

    # Apply crop first (on the original), then pad to target
    (cz0, cz1), (cx0, cx1), (cy0, cy1) = crops
    vol_c = vol_1dhw[:, cz0:cz1, cx0:cx1, cy0:cy1]

    (pz0, pz1), (px0, px1), (py0, py1) = pads
    if any(p > 0 for p in (pz0, pz1, px0, px1, py0, py1)):
        vol_t = F.pad(vol_c, pad=(py0, py1, px0, px1, pz0, pz1), mode=mode)
    else:
        vol_t = vol_c

    spec = {
        "pads": pads,
        "crops": crops,
        "in_shape": in_shape,
        "target_shape": target_shape,
    }
    return vol_t, spec


def invert_pad_crop_flow(flow_t: torch.Tensor, spec) -> torch.Tensor:
    """
    flow_t: (1,3,Td,Th,Tw) predicted at target_shape
    Returns flow mapped back to original crop size (1,3,D,H,W) **without resampling**:
      - if we padded, we slice away pads
      - if we cropped, we zero-embed predicted flow into the crop region
    """
    assert flow_t.ndim == 5 and flow_t.shape[1] == 3
    (D, H, W) = spec["in_shape"]
    (Td, Th, Tw) = spec["target_shape"]
    (pz0, pz1), (px0, px1), (py0, py1) = spec["pads"]
    (cz0, cz1), (cx0, cx1), (cy0, cy1) = spec["crops"]

    # 1) Remove any padding by slicing on the flow prediction
    z0_src = pz0
    z1_src = Td - pz1
    x0_src = px0
    x1_src = Th - px1
    y0_src = py0
    y1_src = Tw - py1
    flow_after_unpad = flow_t[:, :, z0_src:z1_src, x0_src:x1_src, y0_src:y1_src]

    # 2) If original input was cropped to target, zero-embed into original crop size
    out = torch.zeros((1, 3, D, H, W), device=flow_t.device, dtype=flow_t.dtype)
    # region in the original crop where the model actually saw data:
    #   if we cropped, (cz0:cz1) is where the target sat; if we padded, cz0=0,cz1=D etc.
    out[:, :, cz0:cz1, cx0:cx1, cy0:cy1] = flow_after_unpad
    return out
# ------------------------------------------------------------------------------


@torch.no_grad()
def predict_flow_with_padcrop(model, ref_cropN: torch.Tensor, mov_cropN: torch.Tensor, target_shape=(64, 352, 352)):
    """
    ref_cropN, mov_cropN: (1,Dc,Hc,Wc) normalized, cropped to same shape
    1) center pad/crop to 'target_shape' (NO resampling)
    2) run model
    3) invert pad/crop to original crop size (NO resampling)
    returns: flow at crop size (1,3,dc,hc,wc)
    """
    # compute transform spec once (both have same shape)
    ref_in, spec = pad_or_crop_volume(ref_cropN, target_shape, mode="reflect")
    mov_in, _   = pad_or_crop_volume(mov_cropN, target_shape, mode="reflect")

    # add batch + channel dims expected by your model (B,1,D,H,W)
    ref_in = ref_in.unsqueeze(0)
    mov_in = mov_in.unsqueeze(0)

    flow_low, flow_pr = model(ref_in, mov_in, test_mode=1)
    flow_t = flow_pr[-1] if isinstance(flow_pr, (list, tuple)) else flow_pr  # (1,3,Td,Th,Tw)

    # invert pad/crop: back to (1,3,Dc,Hc,Wc)
    flow_crop = invert_pad_crop_flow(flow_t, spec)
    return flow_crop


def main():
    # ------------------ paths & config ------------------
    #root_dir = r"D:\Data\PhD\projects\liver_reg_test\test_data\meas_MID00116_FID06576_vibe963_xd_sweep_32_acc0p3_bin8"
    root_dir = r"D:\ivibe\training\meas_MID00079_FID10494_vibe963_xd_sweep_16_acc0p9_bin8"
    img_path = os.path.join(root_dir, "reconstructed_img_fista_reassinged_direct_abs.mhd")

    # network + checkpoint config
    cwd = os.getcwd()
    if cwd == "/code":
        cwd = "/z0043wnf/GMRAFT/"
    config_path = os.path.join(cwd, "configs", "test_vibe_pairwise.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    # Pull dataset-ish params (fall back to DataLoader defaults)
    ds_cfg = config.get("dataset", config)  # be permissive
    target_shape = tuple(ds_cfg.get("target_shape", [64, 352, 352]))  # <-- new default bigger training grid
    crop_margin = tuple(ds_cfg.get("crop_margin", [8, 16, 16]))
    morph_iter = int(ds_cfg.get("morph_iter", 2))
    min_foreground_q = float(ds_cfg.get("min_foreground_q", 0.20))
    k_mad = float(ds_cfg.get("k_mad", 15.0))

    checkpoint_path = os.path.join(cwd, config["trainer"]["save_dir"], config["name"], "142_final.pth")

    # ------------------ load stack ------------------
    img_np = np.abs(MetaImageIO.read(img_path)).squeeze()  # (states, D, H, W)
    assert img_np.ndim == 4, f"Expected (T,D,H,W), got {img_np.shape}"
    num_states, D, H, W = img_np.shape
    print(f"[info] stack loaded: states={num_states}, D={D}, H={H}, W={W}")

    # torch versions
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stack_t = torch.from_numpy(img_np).float().to(device)  # (T,D,H,W)

    # Per-frame normalization (exactly as training)
    stack_norm = normalize(stack_t)  # (T,D,H,W)

    # Compute union bbox like DataLoader (on normalized stack)
    bbox = bbox_from_volume_union_like_loader(
        stack_norm, k_mad=k_mad, min_foreground_q=min_foreground_q,
        morph_iter=morph_iter, crop_margin=crop_margin
    )
    (z0, z1), (x0, x1), (y0, y1) = bbox
    print(f"[info] bbox (z:{z0}-{z1}, x:{x0}-{x1}, y:{y0}-{y1})")

    # Fixed state
    fixed_full = stack_t[0].unsqueeze(0)             # (1,D,H,W)
    fixed_crop = crop_zhy(fixed_full, bbox)          # (1,dc,hc,wc)
    fixed_cropN = normalize(fixed_crop)              # (1,dc,hc,wc)

    # ------------------ load network ------------------
    model = GMARAFT_Denoiser3D().to(device).eval()
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    print(f"[info] model loaded: {checkpoint_path}")

    # ------------------ outputs ------------------
    registered_volume = np.zeros_like(img_np, dtype=np.float32)                  # (T,D,H,W)
    deformation_fields_fwd = np.zeros((num_states, D, H, W, 3), np.float32)      # voxel disp
    deformation_fields_inv = np.zeros((num_states, D, H, W, 3), np.float32)

    # state 0: identity
    registered_volume[0] = img_np[0]
    deformation_fields_fwd[0] = 0.0
    deformation_fields_inv[0] = 0.0

    # ------------------ loop over motion states ------------------
    for i in range(0, num_states):
        print(f"[run] Registering motion state {i+1}/{num_states} ...")

        moving_full = stack_t[i].unsqueeze(0)                 # (1,D,H,W)
        moving_crop = crop_zhy(moving_full, bbox)             # (1,dc,hc,wc)
        moving_cropN = normalize(moving_crop)

        # Predict flow using ONLY pad/crop to target grid (no resampling)
        flow_crop = predict_flow_with_padcrop(
            model, fixed_cropN, moving_cropN, target_shape=target_shape
        )  # (1,3,dc,hc,wc)

        # Paste flow into full-FOV
        flow_full = paste_flow((D, H, W), flow_crop, bbox)  # (1,3,D,H,W)

        # Warp full-FOV moving volume (note channel order swap kept as in your code)
        warped = warp_3d_batch_bspline(moving_full.unsqueeze(0), flow_full[:, [2, 1, 0], ...])  # (1,1,D,H,W)
        registered_volume[i] = warped.squeeze().detach().cpu().numpy()

        # Save forward field (D,H,W,3)
        flow_vox_dhwd3 = flow_full[0].permute(1, 2, 3, 0).detach().cpu().numpy()
        deformation_fields_fwd[i] = flow_vox_dhwd3

        # Inversion (optional; left commented as in your script)
        # flow_for_inv = torch.from_numpy(flow_vox_dhwd3[None, ...]).to(device).float()  # (1,D,H,W,3)
        # inv_field = invert_displacement_fixed_point_3d(flow_for_inv, num_iters=40)
        # validate_inverse_field(flow_for_inv, inv_field, pad=2)
        # deformation_fields_inv[i] = inv_field.squeeze(0).detach().cpu().numpy()

        torch.cuda.empty_cache()

    # ------------------ save outputs ------------------
    reg_path = os.path.join(root_dir, "registered_network.mhd")
    fwd_path = os.path.join(root_dir, "deformation_fields_fwd_network.mhd")
    inv_path = os.path.join(root_dir, "deformation_fields_inv_network.mhd")

    MetaImageIO.write(reg_path, registered_volume)          # (states,D,H,W)
    MetaImageIO.write(fwd_path, deformation_fields_fwd)     # (states,D,H,W,3)
    MetaImageIO.write(inv_path, deformation_fields_inv)     # (states,D,H,W,3)

    print("\n✅ Saved:")
    print("  → Registered Volume         :", reg_path)
    print("  → Forward Deformation Field :", fwd_path)
    # print("  → Inverse Deformation Field :", inv_path)

    # ------------------ simple stats ------------------
    fwd_mag = np.linalg.norm(deformation_fields_fwd, axis=-1)  # (states,D,H,W)
    inv_mag = np.linalg.norm(deformation_fields_inv, axis=-1)

    print("\n🔍 Forward field stats:")
    print(f"   shape={deformation_fields_fwd.shape}, "
          f"min={fwd_mag.min():.4f}, max={fwd_mag.max():.4f}, mean={fwd_mag.mean():.4f}")
    print("🔍 Inverse field stats:")
    print(f"   shape={deformation_fields_inv.shape}, "
          f"min={inv_mag.min():.4f}, max={inv_mag.max():.4f}, mean={inv_mag.mean():.4f}")

    # set to True if you want plots
    if False:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        plt.title("Histogram of forward deformation magnitudes (vox)")
        plt.hist(fwd_mag.flatten(), bins=100, range=(0, 20))
        plt.xlabel("Displacement (vox)")
        plt.ylabel("Voxel count")

        plt.subplot(1,2,2)
        plt.title("Histogram of inverse deformation magnitudes (vox)")
        plt.hist(inv_mag.flatten(), bins=100, range=(0, 20))
        plt.xlabel("Displacement (vox)")
        plt.ylabel("Voxel count")

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
