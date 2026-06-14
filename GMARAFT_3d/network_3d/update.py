__author__ = "Semih Tarik Uenal"

import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention3d import Aggregate3D,WindowAggregate3D


class FlowHead3D(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead3D, self).__init__()
        self.conv1 = nn.Conv3d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv3d(hidden_dim, 3, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))
    
class SepConvGRU3D_P3D(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super().__init__()
        self.convz1 = nn.Conv3d(hidden_dim + input_dim, hidden_dim,
                                kernel_size=(5,1,1), padding=(2,0,0))
        self.convr1 = nn.Conv3d(hidden_dim + input_dim, hidden_dim,
                                kernel_size=(5,1,1), padding=(2,0,0))
        self.convq1 = nn.Conv3d(hidden_dim + input_dim, hidden_dim,
                                kernel_size=(5,1,1), padding=(2,0,0))

        self.convz2 = nn.Conv3d(hidden_dim + input_dim, hidden_dim,
                                kernel_size=(1,5,5), padding=(0,2,2))
        self.convr2 = nn.Conv3d(hidden_dim + input_dim, hidden_dim,
                                kernel_size=(1,5,5), padding=(0,2,2))
        self.convq2 = nn.Conv3d(hidden_dim + input_dim, hidden_dim,
                                kernel_size=(1,5,5), padding=(0,2,2))

    def forward(self, h, x):
        # h: [B, hidden_dim, D, H, W]
        # x: [B, input_dim,  D, H, W]


        hx = torch.cat([h, x], dim=1)
        z  = torch.sigmoid(self.convz1(hx))
        r  = torch.sigmoid(self.convr1(hx))
        q  = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))
        h1 = (1 - z) * h + z * q


        hx = torch.cat([h1, x], dim=1)
        z  = torch.sigmoid(self.convz2(hx))
        r  = torch.sigmoid(self.convr2(hx))
        q  = torch.tanh(self.convq2(torch.cat([r*h1, x], dim=1)))
        h2 = (1 - z) * h1 + z * q

        return h2

class SepConvGRU3D(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(SepConvGRU3D, self).__init__()
        self.convz1 = nn.Conv3d(hidden_dim+input_dim, hidden_dim, (1,1,5), padding=(0,0,2))
        self.convr1 = nn.Conv3d(hidden_dim+input_dim, hidden_dim, (1,1,5), padding=(0,0,2))
        self.convq1 = nn.Conv3d(hidden_dim+input_dim, hidden_dim, (1,1,5), padding=(0,0,2))

        self.convz2 = nn.Conv3d(hidden_dim+input_dim, hidden_dim, (1,5,1), padding=(0,2,0))
        self.convr2 = nn.Conv3d(hidden_dim+input_dim, hidden_dim, (1,5,1), padding=(0,2,0))
        self.convq2 = nn.Conv3d(hidden_dim+input_dim, hidden_dim, (1,5,1), padding=(0,2,0))

        self.convz3 = nn.Conv3d(hidden_dim+input_dim, hidden_dim, (5,1,1), padding=(2,0,0))
        self.convr3 = nn.Conv3d(hidden_dim+input_dim, hidden_dim, (5,1,1), padding=(2,0,0))
        self.convq3 = nn.Conv3d(hidden_dim+input_dim, hidden_dim, (5,1,1), padding=(2,0,0))

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q

        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q

        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz3(hx))
        r = torch.sigmoid(self.convr3(hx))
        q = torch.tanh(self.convq3(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q

        return h



class SepConvGRU3D_P3D(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):

        self.convz1 = nn.Conv3d(hidden_dim + input_dim, hidden_dim,
                                kernel_size=(3,1,1), padding=(1,0,0))
        self.convr1 = nn.Conv3d(hidden_dim + input_dim, hidden_dim,
                                kernel_size=(3,1,1), padding=(1,0,0))
        self.convq1 = nn.Conv3d(hidden_dim + input_dim, hidden_dim,
                                kernel_size=(3,1,1), padding=(1,0,0))
        
        self.convz2 = nn.Conv3d(hidden_dim + input_dim, hidden_dim,
                                kernel_size=(1,3,3), padding=(0,1,1))
        self.convr2 = nn.Conv3d(hidden_dim + input_dim, hidden_dim,
                                kernel_size=(1,3,3), padding=(0,1,1))
        self.convq2 = nn.Conv3d(hidden_dim + input_dim, hidden_dim,
                                kernel_size=(1,3,3), padding=(0,1,1))

        self.norm1 = nn.GroupNorm(8, hidden_dim)
        self.norm2 = nn.GroupNorm(8, hidden_dim)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q
        h = self.norm1(h)


        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q
        h = self.norm2(h)

        return h

class BasicMotionEncoder3D(nn.Module):
    def __init__(self, corr_levels, corr_radius):
        super(BasicMotionEncoder3D, self).__init__()
        cor_planes = corr_levels * (2*corr_radius + 1)**3
        self.convc1 = nn.Conv3d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv3d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv3d(3, 128, 7, padding=3)
        self.convf2 = nn.Conv3d(128, 64, 3, padding=1)
        self.conv = nn.Conv3d(64+192, 128-3, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)



class GMAUpdateBlock3D(nn.Module):
    def __init__(self, corr_levels=3, corr_radius=3, hidden_dim=128, factor=4, num_heads=1):
        super().__init__()
        self.encoder = BasicMotionEncoder3D(corr_levels, corr_radius)
        self.gru = SepConvGRU3D(hidden_dim=hidden_dim, input_dim=128 + hidden_dim + hidden_dim)
        self.flow_head = FlowHead3D(hidden_dim, hidden_dim=256)

        self.mask = nn.Sequential(
            nn.Conv3d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, factor * factor * factor* 27, 1, padding=0)  # 27 = 3×3×3 kernel
        )

        self.aggregator = Aggregate3D(dim=128, dim_head=128, heads=num_heads)

    def forward(self, net, inp, corr, flow, attention):

        motion_features = self.encoder(flow, corr)
        motion_features_global = self.aggregator(attention, motion_features)
        inp_cat = torch.cat([inp, motion_features, motion_features_global], dim=1)

        net = self.gru(net, inp_cat)
        delta_flow = self.flow_head(net)
        mask = 0.25 * self.mask(net)

        return net, mask, delta_flow
