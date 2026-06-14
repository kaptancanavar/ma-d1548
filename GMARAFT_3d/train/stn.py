__author__ = "Semih Tarik Uenal"

import torch
import torch.nn.functional as F
from einops import rearrange

class SpatialTransformer(torch.nn.Module):
    def __init__(self):
        super(SpatialTransformer, self).__init__()

    def forward(self, src, flow, mode='bilinear'):
        if src.dim() == 5:
            [b_1, f_1, c_1, h_1, w_1] = src.shape
            src = torch.reshape(src, (b_1 * f_1, c_1, h_1, w_1))
        if flow.dim() == 5:
            [b, f, c, h, w] = flow.shape
            flow = torch.reshape(flow, (b * f, c, h, w))
            reshape = True
        else:
            reshape = False

        shape = flow.shape[2:]

        vectors = [torch.arange(0, s) for s in shape]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)  # y, x, z
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)
        grid = grid.cuda()
        # grid = grid

        new_locs = grid + flow

        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]

        out = F.grid_sample(src, new_locs, mode=mode)

        if reshape:
            out = rearrange(out, '(b f) c h w -> b f c h w', b=b)

        return out
    



class SpatialTransformer3D(torch.nn.Module):
    def __init__(self):
        super(SpatialTransformer3D, self).__init__()

    def forward(self, src, flow, mode='bilinear'):
        if src.dim() == 6:
            b_1, f_1, c_1, d_1, h_1, w_1 = src.shape
            src = src.view(b_1 * f_1, c_1, d_1, h_1, w_1)
        if flow.dim() == 6:
            b, f, c, d, h, w = flow.shape
            flow = flow.view(b * f, c, d, h, w)
            reshape = True
        else:
            reshape = False

        D, H, W = flow.shape[2:]
        device = flow.device

        z = torch.linspace(0, D - 1, D, device=device)
        y = torch.linspace(0, H - 1, H, device=device)
        x = torch.linspace(0, W - 1, W, device=device)
        zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij') 
        grid = torch.stack((xx, yy, zz), dim=0).unsqueeze(0).repeat(flow.shape[0], 1, 1, 1, 1)  # [Bf, 3, D, H, W]

        new_locs = grid + flow 
        new_locs[:, 0] = 2.0 * (new_locs[:, 0] / (W - 1) - 0.5) 
        new_locs[:, 1] = 2.0 * (new_locs[:, 1] / (H - 1) - 0.5)  
        new_locs[:, 2] = 2.0 * (new_locs[:, 2] / (D - 1) - 0.5)  


        new_locs = new_locs.permute(0, 2, 3, 4, 1) #[Bf, D, H, W, 3]

        out = F.grid_sample(src, new_locs, mode=mode, align_corners=True, padding_mode='border')

        if reshape:
            out = rearrange(out, '(b f) c d h w -> b f c d h w', b=b)

        return out