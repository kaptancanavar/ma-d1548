__author__ = "Semih Tarik Uenal"

import torch
import torch.nn.functional as F
from .utils import bspline_sampler,trilinear_sampler  

try:
    import alt_cuda_corr
except:
    pass


class CorrBlock3D:
    def __init__(self, fmap1, fmap2, num_levels=3, radius=6):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        corr = CorrBlock3D.corr(fmap1, fmap2)  # (B, D, H, W, C, D, H, W)
        B, D1, H1, W1, C, D2, H2, W2 = corr.shape

        corr = corr.view(B * D1 * H1 * W1, C, D2, H2, W2)
        self.corr_pyramid.append(corr)

        for _ in range(self.num_levels - 1):
            corr = F.avg_pool3d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 4, 1)  # (B, D, H, W, 3)

        B, D1, H1, W1, _ = coords.shape
        out_pyramid = []

        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]


            dx = torch.linspace(-r, r, 2*r+1, device=coords.device)
            dy = torch.linspace(-r, r, 2*r+1, device=coords.device)
            dz = torch.linspace(-r, r, 2*r+1, device=coords.device)
            delta = torch.stack(torch.meshgrid(dx, dy, dz, indexing='ij'), dim=-1)


            centroid_lvl = coords.reshape(B * D1 * H1 * W1, 1, 1, 1, 3) / (2 ** i)
            delta_lvl = delta.view(1, *delta.shape[:3], 3)
            coords_lvl = centroid_lvl + delta_lvl  # (B*D1*H1*W1, Z, Y, X, 3)
            corr_sampled = trilinear_sampler(corr, coords_lvl)
 
            corr_sampled = corr_sampled.view(B, D1, H1, W1, -1)
            out_pyramid.append(corr_sampled)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 4, 1, 2, 3).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        B, C, D, H, W = fmap1.shape
        fmap1 = fmap1.view(B, C, D * H * W)
        fmap2 = fmap2.view(B, C, D * H * W)

        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)  # (B, N, N), N = D*H*W
        corr = corr.view(B, D, H, W, 1, D, H, W)
        return corr / torch.sqrt(torch.tensor(C, dtype=torch.float32, device=fmap1.device))
