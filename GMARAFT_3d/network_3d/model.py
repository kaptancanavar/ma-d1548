__author__ = "Semih Tarik Uenal"

import torch
import torch.nn as nn
import torch.nn.functional as F
from .update import GMAUpdateBlock3D
from .encoder3d import BasicEncoder3D
from .corr3d import CorrBlock3D
from .attention3d import Attention3D,WindowAttention3D
from .utils import coords_grid_3d, upflow4_3d

autocast = torch.cuda.amp.autocast

class GMARAFT_Denoiser3D(nn.Module):
    def __init__(self):
        super().__init__()

        self.num_channels = 1
        self.num_heads    = 1
        self.iters        = 4
        self.corr_radius  = 3
        self.hidden_dim   = 128
        self.context_dim  = 128

        self.fnet = BasicEncoder3D(
            output_dim=256, norm_fn='instance', num_input_channels=self.num_channels
        )
        self.cnet = BasicEncoder3D(
            output_dim=self.hidden_dim + self.context_dim,
            norm_fn='group',
            num_input_channels=1
        )

        self.att = Attention3D(dim=self.context_dim,
                                        heads=self.num_heads,
                                        dim_head=self.context_dim)
        self.update_block = GMAUpdateBlock3D(hidden_dim=self.hidden_dim)

    def initialize_flow(self, img, factor=4):
        B, C, D, H, W = img.shape
        coords0 = coords_grid_3d(B, D // factor, H // factor, W // factor).to(img.device)
        coords1 = coords_grid_3d(B, D // factor, H // factor, W // factor).to(img.device)
        return coords0, coords1

    def forward(self, image1, image2, flow_init=None, test_mode=False):
        B, C, D, H, W = image1.shape
        # image1 = F.interpolate(image1, scale_factor=(0.5, 0.5, 0.5), mode='trilinear', align_corners=True)
        # image2 = F.interpolate(image2, scale_factor=(0.5, 0.5, 0.5), mode='trilinear', align_corners=True)

        image1 = 2*(image1) - 1.0
        image2 = 2*(image2) - 1.0
        image1, image2 = image1.contiguous(), image2.contiguous()

        with autocast():
            fmap1, fmap2 = self.fnet([image1, image2])
        fmap1, fmap2 = fmap1.float(), fmap2.float()
        corr_fn = CorrBlock3D(fmap1, fmap2, radius=self.corr_radius)

        with autocast():
            cnet = self.cnet(image2)
            net, inp = torch.split(cnet, [self.hidden_dim, self.context_dim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)
            attention = self.att(inp)

        coords0, coords1 = self.initialize_flow(image1)
        if flow_init is not None:
            fd, fh, fw = flow_init.shape[-3:]
            gd, gh, gw = coords1.shape[-3:]
            flow_init = F.interpolate(flow_init, size=(gd, gh, gw), mode='trilinear', align_corners=False)
            flow_init[:, 0] *= gd / float(fd)
            flow_init[:, 1] *= gh / float(fh)
            flow_init[:, 2] *= gw / float(fw)
            coords1 = coords1 + flow_init

        flow_predictions = []
        for _ in range(self.iters):
            coords1 = coords1.detach()
            corr    = corr_fn(coords1)
            flow    = coords1 - coords0

            with autocast():
                net, up_mask, delta_flow = self.update_block(
                    net, inp, corr, flow, attention
                )
            coords1 = coords1 + delta_flow

            if up_mask is None:
                flow_up = upflow4_3d(coords1 - coords0)
            else:
                flow_up = self.upsample_flow_3d(coords1 - coords0, up_mask)
            flow_predictions.append(flow_up)
            # flow_predictions.append(self.upsample_flow_doubleSize(flow_up))

        if test_mode:
            return coords1 - coords0, flow_predictions[-1]


        return flow_predictions

    def upsample_flow_3d(self, flow, mask):
        N, C, D, H, W = flow.shape
        flow4 = flow * 4.0

        mask = mask.view(N, 1, 27, 4, 4, 4, D, H, W)
        mask = torch.softmax(mask, dim=2)

        weight = flow.new_zeros(C*27, 1, 3, 3, 3)
        for k in range(27):
            dz, rem = divmod(k, 9)
            dy, dx  = divmod(rem, 3)
            weight[k::27, 0, dz, dy, dx] = 1.0
        patches = F.conv3d(flow4, weight, padding=1, groups=C)
        patches = patches.view(N, C, 27, D, H, W)

        out = (mask * patches.unsqueeze(3).unsqueeze(4).unsqueeze(5)).sum(dim=2)
        out = out.permute(0,1,5,2,6,3,7,4).contiguous()
        return out.view(N, C, 4*D, 4*H, 4*W)

    def upsample_flow_doubleSize(self, flow):
        N, C, D_in, H_in, W_in = flow.shape
        new_size = (2 * D_in, 2 * H_in, 2 * W_in)
        up = F.interpolate(flow, size=new_size, mode='trilinear', align_corners=True) * 2
        return up