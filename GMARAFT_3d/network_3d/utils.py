__author__ = "Semih Tarik Uenal"

import torch
import torch.nn.functional as F

try:
    from train.warp import warp_3d_batch_bspline
except Exception:
    warp_3d_batch_bspline = None

try:
    import bspline_interp_withgrad
except ImportError:
    bspline_interp_withgrad = None


def coords_grid_3d(batch, depth, height, width, device=None):
    """ Create a meshgrid of pixel coordinates [B, 3, D, H, W] """
    coords = torch.meshgrid(
        torch.arange(depth, device=device),
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing='ij'
    )
    coords = torch.stack(coords[::-1], dim=0).float()  # (3, D, H, W)
    return coords[None].expand(batch, -1, -1, -1, -1)  # (B, 3, D, H, W)


def upflow4_3d(flow, mode='trilinear'):
    """ Upsample 3D flow by factor 4 along D, H, W axes """
    D, H, W = flow.shape[2:]
    new_size = (4 * D, 4 * H, 4 * W)
    return 4 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)


def trilinear_sampler(img, coords, mask=False):
    """
    Wrapper for 3D grid_sample using voxel coordinates.
    img: [B, C, D, H, W]
    coords: [B, D, H, W, 3] in voxel units
    """
    B, C, D, H, W = img.shape
    xgrid, ygrid, zgrid = coords.split([1, 1, 1], dim=-1)  # [B, D, H, W, 1] each

    xgrid = 2.0 * xgrid / (W - 1) - 1.0
    ygrid = 2.0 * ygrid / (H - 1) - 1.0
    zgrid = 2.0 * zgrid / (D - 1) - 1.0

    grid = torch.cat([xgrid, ygrid, zgrid], dim=-1)  # [B, D, H, W, 3]

    sampled = F.grid_sample(img, grid, mode='bilinear', align_corners=True, padding_mode='border')

    if mask:
        valid = (xgrid > -1) & (xgrid < 1) & \
                (ygrid > -1) & (ygrid < 1) & \
                (zgrid > -1) & (zgrid < 1)
        return sampled, valid.float()

    return sampled
def pad_image_for_bspline(img: torch.Tensor, padding: int = 2) -> torch.Tensor:
    """
    Pad (Z, Y, X) axes symmetrically for cubic B-spline support.
    """
    return torch.nn.functional.pad(img, pad=(padding, padding, padding, padding, padding, padding), mode='reflect')


def bspline_sampler(img: torch.Tensor,
                    coords: torch.Tensor,
                    pad: int = 2,
                    mask: bool = False):
    
    if bspline_interp_withgrad is None:
        raise ImportError("bspline_interp_withgrad not installed. Use trilinear_sampler/grid_sample.")

    """
    Wrapper for 3D B-spline warp using voxel coordinates.
    img:    [B, C, D, H, W]
    coords: [B, D, H, W, 3] in voxel units
    """
    B, C, D, H, W = img.shape

    xgrid, ygrid, zgrid = coords.split([1, 1, 1], dim=-1)  # each [B, D, H, W, 1]
    coords = torch.cat([xgrid, ygrid, zgrid], dim=-1)      # [B, D, H, W, 3]

    M = B * C
    img_flat = img.view(M, D, H, W)

    # pad + prefilter
    img_padded = F.pad(img_flat,
                       (pad, pad, pad, pad, pad, pad),
                       mode='reflect')
    img_prefiltered = bspline_interp_withgrad.bspline_prefilter_autograd(
        img_padded.contiguous()
    )

    # expand coords over channels and flatten
    coords_flat = (
        coords
        .unsqueeze(1)                   # [B,1,D,H,W,3]
        .expand(B, C, D, H, W, 3)       # [B,C,D,H,W,3]
        .contiguous()
        .view(M, D, H, W, 3)            # [M,D,H,W,3]
    )
    coords_flat[..., 0] += pad  # z
    coords_flat[..., 1] += pad  # y
    coords_flat[..., 2] += pad  # x

    warped_flat = bspline_interp_withgrad.bspline_interp_autograd(
        img_prefiltered,
        coords_flat.contiguous()
    )  # [M, D, H, W]

    warped = warped_flat.view(B, C, D, H, W)

    if mask:
        valid = (
            (xgrid > -pad) & (xgrid < W + pad) &
            (ygrid > -pad) & (ygrid < H + pad) &
            (zgrid > -pad) & (zgrid < D + pad)
        ).unsqueeze(1).float()  # [B,1,D,H,W]
        return warped, valid

    return warped