__author__ = "Semih Tarik Uenal"

import torch
import torch.nn as nn
import torch.nn.functional as F
from .update import GMAUpdateBlock3D
from .encoder3d import BasicEncoder3D
from .corr3d import CorrBlock3D
from .attention3d import Attention3D, WindowAttention3D
from .denoiser3d import ResNet3D
from .utils import coords_grid_3d, upflow4_3d

autocast = torch.cuda.amp.autocast

class RAFTBase3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_channels = 1
        self.num_heads = 1
        self.iters = 6
        self.corr_radius = 3
        self.hidden_dim = 128
        self.context_dim = 128

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm3d):
                m.eval()




    def initialize_flow(self, img, factor=4):
        B, C, D, H, W = img.shape
        coords0 = coords_grid_3d(B, D // factor, H // factor, W // factor).to(img.device)
        coords1 = coords_grid_3d(B, D // factor, H // factor, W // factor).to(img.device)
        return coords0, coords1
    
    def unfold3d(self, x, kernel_size=3, padding=1):
        """Extracts 3D local patches like F.unfold does for 2D."""
        B, C, D, H, W = x.shape
        x = F.pad(x, (padding, padding, padding, padding, padding, padding))  # pad W, H, D
        patches = x.unfold(2, kernel_size, 1).unfold(3, kernel_size, 1).unfold(4, kernel_size, 1)
        # shape: [B, C, D, H, W, k, k, k]
        patches = patches.contiguous().view(B, C, D, H, W, -1)
        return patches.permute(0, 1, 5, 2, 3, 4)  # [B, C, 27, D, H, W]

    def upsample_flow(self,flow, mask):
        """
        Learnable trilinear upsampling of 3D flow field.
        flow: [B, 3, D, H, W]
        mask: [B, 1, 27*64, D, H, W] → [B, 1, 27, 4, 4, 4, D, H, W]
        """
        B, _, D, H, W = flow.shape
        mask = mask.view(B, 1, 27, 4, 4, 4, D, H, W)  # [B, 1, 27, 4, 4, 4, D, H, W]
        flow = 4.0 * flow
        mask = torch.softmax(mask, dim=2)

        flow_patches = self.unfold3d(flow, kernel_size=3, padding=1)

        flow_patches = flow_patches.unsqueeze(3).unsqueeze(4).unsqueeze(5)
        up_flow = torch.sum(mask * flow_patches, dim=2)  # [B, 3, 1, 1, 1, D, H, W]
        up_flow = up_flow.view(B, 3, 4, 4, 4, D, H, W)    # [B, 3, 4, 4, 4, D, H, W]
        up_flow = up_flow.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()  # [B, 3, D, 4, H, 4, W, 4]
        up_flow = up_flow.view(B, 3, D * 4, H * 4, W * 4)

        return up_flow
    
    def upsample_flow22(self, flow, mask):
        """
        Args:
            flow: Tensor[N, C, D, H, W]        - coarse flow, maybe in fp16
            mask: Tensor[N, 27*64, D, H, W]    - upsampling masks in fp32
        Returns:
            out:  Tensor[N, C, 4*D, 4*H, 4*W]  - upsampled flow
        """
        N, C, D, H, W = flow.shape

        m = (
            mask
            .to(flow.dtype)                          # cast to flow.dtype
            .view(N, 27, 64, D, H, W)
            .softmax(dim=1)
            .view(N, 27, 4, 4, 4, D, H, W)
        )

        flow4 = flow * 4.0

        #  one 3×3×3 kernel per neighbor, repeated for each channel
        weight = flow.new_zeros(C * 27, 1, 3, 3, 3)
        for k in range(27):
            dz, rem = divmod(k, 9)
            dy, dx = divmod(rem, 3)
            weight[k::27, 0, dz, dy, dx] = 1.0

        neigh = F.conv3d(flow4, weight, padding=1, groups=C)  # [N, C*27, D, H, W]
        neigh = neigh.view(N, C, 27, D, H, W)                # [N, C, 27, D, H, W]

        # convex combination with the mask
        #    neigh: [N, C, 27, D, H, W]
        #    m:     [N, 27, 4, 4, 4, D, H, W]
        #    out:   [N, C, 4, 4, 4, D, H, W]
        out = torch.einsum('nckdhw,nkijldhw->nciljdhw', neigh, m)

        out = out.permute(0, 1, 5, 2, 6, 3, 7, 4)
        out = out.reshape(N, C, 4 * D, 4 * H, 4 * W)

        return out

    # def upsample_flow_3d(self,flow, mask):
    #     N, C, D, H, W = flow.shape
    #     m = mask.view(N,27,64,D,H,W).softmax(dim=2).view(N,27,4,4,4,D,H,W)
    #     flow4 = flow * 4.0
    #     weight = torch.zeros(C*27,1,3,3,3,device=flow.device)
    #     for k in range(27):
    #         dz, rem = divmod(k,9)
    #         dy, dx = divmod(rem,3)
    #         weight[k::27,0,dz,dy,dx] = 1.0
    #     neigh = F.conv3d(flow4, weight, padding=1, groups=C).view(N,C,27,D,H,W)
    #     dtype = neigh.dtype
    #     m = m.to(dtype)
    #     out = torch.einsum('nckdhw,nkijldhw->nciljdhw', neigh, m)
    #     out = out.permute(0,1,5,2,6,3,7,4)  # [N,C,4D,4H,4W]
    #     return out.reshape(N,C,4*D,4*H,4*W)
    

    def upsample_flow_3d(self, flow, mask):
        """
        3D upsampling of flow field using learned convex combination.
        Args:
            flow: (N, C, D, H, W)
            mask: (N, 27, 64, D, H, W)
        Returns:
            Upsampled flow: (N, C, 4*D, 4*H, 4*W)
        """
        N, C, D, H, W = flow.shape
        flow4 = flow * 4.0

        mask = mask.view(N, 1, 27, 4, 4, 4, D, H, W)
        mask = torch.softmax(mask, dim=2)

        weight = torch.zeros(C * 27, 1, 3, 3, 3, device=flow.device)
        for k in range(27):
            dz, rem = divmod(k, 9)
            dy, dx = divmod(rem, 3)
            weight[k::27, 0, dz, dy, dx] = 1.0
        patches = F.conv3d(flow4, weight, padding=1, groups=C).view(N, C, 27, D, H, W)

        out = (mask * patches[:, :, :, None, None, None, :, :, :]).sum(dim=2)

        out = out.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        return out.view(N, C, 4 * D, 4 * H, 4 * W)



class GMARAFT_Denoiser3D(RAFTBase3D):
    def __init__(self):
        super(GMARAFT_Denoiser3D, self).__init__()
        self.fnet = BasicEncoder3D(output_dim=256, norm_fn='instance', num_input_channels=self.num_channels)
        self.cnet = BasicEncoder3D(output_dim=self.hidden_dim + self.context_dim, norm_fn='batch', num_input_channels=3)
        self.update_block = GMAUpdateBlock3D(hidden_dim=self.hidden_dim)
        self.att = Attention3D(dim=self.context_dim, heads=self.num_heads, dim_head=self.context_dim)
        self.resnet = ResNet3D(in_channels=3)

    def forward(self, image1, image2, context_image, flow_init=None, test_mode=False):
        image1 = 2 * (image1) - 1.0
        image2 = 2 * (image2) - 1.0
        context_image = 2 * (context_image) - 1.0

        image1, image2 = image1.contiguous(), image2.contiguous()
        context_image = context_image.contiguous()

        hdim, cdim = self.hidden_dim, self.context_dim

        with autocast(enabled=True):
            fmap1, fmap2 = self.fnet([image1, image2])

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        corr_fn = CorrBlock3D(fmap1, fmap2, radius=self.corr_radius)

        with autocast(enabled=True):
            context_image_up = self.resnet(context_image)
            context_image_up = 2 * ((context_image_up - context_image_up.min()) / (context_image_up.max() - context_image_up.min())) - 1
            cnet = self.cnet(context_image_up)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)
            attention = self.att(inp)

        coords0, coords1 = self.initialize_flow(image1)
        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for _ in range(self.iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)
            flow = coords1 - coords0
            
            with autocast(enabled=True):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow, attention)

            coords1 = coords1 + delta_flow
            
            if up_mask is None:

                # diff = coords1 - coords0
                # print(diff.min(),diff.max())
                flow_up = upflow4_3d(coords1 - coords0)
                # print(flow_up.min(),flow_up.max())
            else:
                diff = coords1 - coords0
                # print(f"pre min {diff.min()} max {diff.max()}")
                flow_up = self.upsample_flow_3d(coords1 - coords0, up_mask)
                # print(f" after min {flow_up.min()} max {flow_up.max()}")

            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up, context_image_up

        return flow_predictions, context_image_up
