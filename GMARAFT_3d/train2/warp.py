__author__ = "Semih Tarik Uenal"

import torch
from einops import rearrange
import  bspline_interp_withgrad 
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

def warp_torch(x, flo, mode='bilinear'):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, (T), C, H, W] (im2)
    flo: [B, (T) ,2, H, W] flow
    """
    if x.dim() == 5:
        b= x.shape[0]
        x =  rearrange(x, 'b f c h w -> (b f) c h w')
        reshape=True
    else: reshape=False

    if flo.dim() == 5:
        flo =  rearrange(flo, 'b f c h w -> (b f) c h w')

    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    mask = torch.ones(x.size(), dtype=x.dtype)
    if x.is_cuda:
        grid = grid.cuda()
        mask = mask.cuda()

    # flo = torch.flip(flo, dims=[1])
    vgrid = grid + flo

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = torch.nn.functional.grid_sample(x, vgrid, align_corners=True, mode=mode)

    mask = torch.nn.functional.grid_sample(mask, vgrid, align_corners=True, mode=mode)


    mask[mask < 0.9999] = 0
    mask[mask > 0] = 1

    out = output * mask

    if reshape:
        out = rearrange(out, '(b f) c h w -> b f c h w', b=b)

    return out




def warp_3d_torch(x, flo, mode='bilinear'):
    """
    Warp a 3D volume x according to flow flo.
    x:   [B, C, D, H, W]          (or with optional time/frame [B, F, C, D, H, W])
    flo: [B, 3, D, H, W]          (or [B, F, 3, D, H, W])
    """
    reshape = False
    if x.dim() == 6:
        # merge frame dim
        b, f, c, d, h, w = x.shape
        x   = x.view(b*f, c, d, h, w)
        flo = flo.view(b*f, 3, d, h, w)
        reshape = True

    B, C, D, H, W = x.shape

    # 1) build a base grid of pixel coordinates [0..W-1], [0..H-1], [0..D-1]
    #    using your existing helper – this returns [B, 3, D, H, W]
    grid = coords_grid_3d(B, D, H, W, device=x.device)  # (B,3,D,H,W)

    # 2) add the flow (in voxel units) to that grid
    coords = grid + flo                               # (B,3,D,H,W)

    # 3) normalize to [-1,1] for grid_sample:
    #    coords[0] is x (width), coords[1] is y (height), coords[2] is z (depth)
    coords_norm = torch.empty_like(coords)
    coords_norm[:, 0] = 2.0 * coords[:, 0] / (W - 1) - 1.0
    coords_norm[:, 1] = 2.0 * coords[:, 1] / (H - 1) - 1.0
    coords_norm[:, 2] = 2.0 * coords[:, 2] / (D - 1) - 1.0

    # 4) reshape into the (N, D, H, W, 3) format
    #    and swap from (x,y,z) channel order to (x,y,z) last-dim order
    grid_for_sampler = coords_norm.permute(0, 2, 3, 4, 1)  # (B, D, H, W, 3)

    # 5) do the actual sampling
    warped =  torch.nn.functional.grid_sample(
        x, grid_for_sampler,
        mode=mode,
        padding_mode='border',
        align_corners=True
    )

    # 6) restore the frame dim if needed
    if reshape:
        warped = warped.view(b, f, C, D, H, W)

    return warped


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
    img = img.squeeze(0)
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

    return warped_real.unsqueeze(0)

