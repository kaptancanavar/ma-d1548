__author__ = "Semih Tarik Uenal"

import torch
from einops import rearrange
import  bspline_interp_withgrad 
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
    Warp a 3D volume x using a 3D flow field flo.
    x:   [B, (T), C, D, H, W]
    flo: [B, (T), 3, D, H, W]  (ΔZ, ΔY, ΔX)
    Returns: warped x with same shape
    """

    if x.dim() == 6:
        b = x.shape[0]
        x = rearrange(x, 'b f c d h w -> (b f) c d h w')
        flo = rearrange(flo, 'b f c d h w -> (b f) c d h w')
        reshape = True
    else:
        reshape = False

    B, C, D, H, W = x.size()


    z = torch.linspace(0, D - 1, D, device=x.device)
    y = torch.linspace(0, H - 1, H, device=x.device)
    x_ = torch.linspace(0, W - 1, W, device=x.device)
    zz, yy, xx = torch.meshgrid(z, y, x_, indexing='ij')
    grid = torch.stack((xx, yy, zz), dim=0).unsqueeze(0).repeat(B, 1, 1, 1, 1)  # [B, 3, D, H, W]

    vgrid = grid + flo 

    vgrid[:, 0] = 2.0 * vgrid[:, 0] / max(W - 1, 1) - 1.0  # X
    vgrid[:, 1] = 2.0 * vgrid[:, 1] / max(H - 1, 1) - 1.0  # Y
    vgrid[:, 2] = 2.0 * vgrid[:, 2] / max(D - 1, 1) - 1.0  # Z

    
    vgrid = vgrid.permute(0, 2, 3, 4, 1) # [B, D, H, W, 3]


    output = torch.nn.functional.grid_sample(x, vgrid, align_corners=True, mode=mode,padding_mode='border')

    mask = torch.ones_like(x)
    mask = torch.nn.functional.grid_sample(mask, vgrid, align_corners=True, mode=mode,padding_mode='border')
    mask[mask < 0.9999] = 0
    mask[mask > 0] = 1

    out = output * mask

    if reshape:
        out = rearrange(out, '(b f) c d h w -> b f c d h w', b=b)

    return out



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
    flow = flow.flip(-1)  # Now (z,y,x)

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

