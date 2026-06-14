__author__ = "Semih Tarik Uenal"

import torch
import torch.nn.functional as F

def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].expand(batch, -1, -1, -1)

def upflow4(flow, mode='bilinear'):
    new_size = (4 * flow.shape[2], 4 * flow.shape[3])
    return 4 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)

def bilinear_sampler(img, coords, mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    B, C, H, W = img.shape
    xgrid, ygrid = coords.split([1, 1], dim=-1)  # [B, H, W, 1]
    
    xgrid = 2 * xgrid / (W - 1) - 1
    ygrid = 2 * ygrid / (H - 1) - 1
    grid = torch.cat([xgrid, ygrid], dim=-1)  # [B, H, W, 2]

    assert img.shape[0] == grid.shape[0], f"Batch size mismatch: img {img.shape[0]}, grid {grid.shape[0]}"
    sampled = F.grid_sample(img, grid, align_corners=True, mode='bilinear', padding_mode='border')

    if mask:
        valid = (xgrid > -1) & (xgrid < 1) & (ygrid > -1) & (ygrid < 1)
        return sampled, valid.float()

    return sampled
