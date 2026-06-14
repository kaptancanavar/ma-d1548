__author__ = "Semih Tarik Uenal"

import torch
from torch import nn, einsum
from einops import rearrange

class Attention3D(nn.Module):
    def __init__(
        self,
        *,
        dim,
        max_pos_size=100,
        heads=4,
        dim_head=128,
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head

        self.to_qk = nn.Conv3d(dim, inner_dim * 2, 1, bias=False)

    def forward(self, fmap):
        heads, b, c, d, h, w = self.heads, *fmap.shape

        q, k = self.to_qk(fmap).chunk(2, dim=1)

        q, k = map(lambda t: rearrange(t, 'b (h d) z x y -> b h z x y d', h=heads), (q, k))
        q = self.scale * q

        sim = einsum('b h z x y d, b h u v w d -> b h z x y u v w', q, k)

        sim = rearrange(sim, 'b h z x y u v w -> b h (z x y) (u v w)')
        attn = sim.softmax(dim=-1)

        return attn


class Aggregate3D(nn.Module):
    def __init__(
        self,
        dim,
        heads=4,
        dim_head=128,
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head

        self.to_v = nn.Conv3d(dim, inner_dim, 1, bias=False)

        self.gamma = nn.Parameter(torch.zeros(1))

        if dim != inner_dim:
            self.project = nn.Conv3d(inner_dim, dim, 1, bias=False)
        else:
            self.project = None

    def forward(self, attn, fmap):
        heads, b, c, d, h, w = self.heads, *fmap.shape

        v = self.to_v(fmap)
        v = rearrange(v, 'b (h d) z x y -> b h (z x y) d', h=heads)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (z x y) d -> b (h d) z x y', z=d, x=h, y=w)

        if self.project is not None:
            out = self.project(out)

        out = fmap + self.gamma * out
        return out
    
class Attention3D32(nn.Module):
    def __init__(
        self,
        *,
        dim,
        max_pos_size=100,    # accepted for compatibility, but unused
        heads=4,
        dim_head=None,       # accepted for compatibility, but unused
        dropout=0.0
    ):
        super().__init__()
        self.heads = heads

        # We ignore max_pos_size and dim_head here,
        # because nn.MultiheadAttention only needs embed_dim & num_heads.
        self.mha = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True
        )

    def forward(self, fmap):
        """
        fmap: (B, C, D, H, W)
        returns: attn (B, heads, N, N) where N=D*H*W
        """
        B, C, D, H, W = fmap.shape
        N = D * H * W

        # 1) flatten spatial dims → (B, N, C)
        x = fmap.view(B, C, N).permute(0, 2, 1)

        # 2) self-attention
        #    need_weights=True & average_attn_weights=False to get per-head maps
        _, attn = self.mha(
            x, x, x,
            need_weights=True,
            average_attn_weights=False
        )
        # attn: (B, heads, N, N)
        return attn

class WindowAttention3D(nn.Module):
    def __init__(
        self,
        dim,
        window_size=(11,11,11),
        heads=4,
        dim_head=32,
        dropout=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = heads * dim_head
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.to_qkv = nn.Conv3d(dim, inner_dim * 3, kernel_size=1, bias=False)
        self.dropout = nn.Dropout(dropout)

        wd, wh, ww = window_size
        self.relative_bias = nn.Parameter(
            torch.zeros((2*wd-1)*(2*wh-1)*(2*ww-1), heads)
        )
        coords = torch.stack(torch.meshgrid(
            torch.arange(wd), torch.arange(wh), torch.arange(ww), indexing='ij'
        ))  # [3, wd, wh, ww]
        coords_flat = coords.reshape(3, -1).permute(1,0)
        rel = coords_flat[:, None, :] - coords_flat[None, :, :]
        rel += torch.tensor([wd-1, wh-1, ww-1])
        idx = (rel[...,0] * (2*wh-1)*(2*ww-1) + rel[...,1] * (2*ww-1) + rel[...,2]).reshape(-1)
        self.register_buffer('relative_index', idx)

    def forward(self, x):
        # x: [B, C, D, H, W]
        B, C, D, H, W = x.shape
        wd, wh, ww = self.window_size
        h, hd = self.heads, self.dim_head
        x_flat = x.view(B, C, -1).permute(0,2,1)
        x_norm = self.norm(x_flat).permute(0,2,1).view(B, C, D, H, W)
        xw = rearrange(
            x_norm,
            'B C (d wd) (h wh) (w ww) -> (B d h w) (wd wh ww) C',
            wd=wd, wh=wh, ww=ww
        )  # [n, w3, C]

        n = xw.shape[0]
        xw5 = xw.view(n, wd, wh, ww, C).permute(0,4,1,2,3)
        qkv = self.to_qkv(xw5).chunk(3, dim=1)
        q, k, _ = map(lambda t: rearrange(t, 'n (h d) wd wh ww -> n h (wd wh ww) d', h=h), qkv[:2])
        q = q * self.scale
        attn = einsum('n h i d, n h j d -> n h i j', q, k)
        bias = self.relative_bias[self.relative_index].view(wd*wh*ww, wd*wh*ww, h).permute(2,0,1)
        attn = attn + bias.unsqueeze(0)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        return attn

class WindowAttention3D(nn.Module):
    def __init__(
        self,
        dim,
        window_size=(11,11,11),
        heads=4,
        dim_head=32,
        dropout=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = heads * dim_head
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.to_qkv = nn.Conv3d(dim, inner_dim * 3, kernel_size=1, bias=False)
        self.dropout = nn.Dropout(dropout)

        wd, wh, ww = window_size
        self.relative_bias = nn.Parameter(
            torch.zeros((2*wd-1)*(2*wh-1)*(2*ww-1), heads)
        )
        coords = torch.stack(torch.meshgrid(
            torch.arange(wd), torch.arange(wh), torch.arange(ww), indexing='ij'
        ))  # [3, wd, wh, ww]
        coords_flat = coords.reshape(3, -1).permute(1,0)
        rel = coords_flat[:, None, :] - coords_flat[None, :, :]
        rel += torch.tensor([wd-1, wh-1, ww-1])
        idx = (rel[...,0] * (2*wh-1)*(2*ww-1) + rel[...,1] * (2*ww-1) + rel[...,2]).reshape(-1)
        self.register_buffer('relative_index', idx)

    def forward(self, x):
        # x: [B, C, D, H, W]
        B, C, D, H, W = x.shape
        wd, wh, ww = self.window_size
        h, hd = self.heads, self.dim_head

        x_flat = x.view(B, C, -1).permute(0,2,1)
        x_norm = self.norm(x_flat).permute(0,2,1).view(B, C, D, H, W)
        xw = rearrange(
            x_norm,
            'B C (d wd) (h wh) (w ww) -> (B d h w) (wd wh ww) C',
            wd=wd, wh=wh, ww=ww
        )  # [n, w3, C]

        n = xw.shape[0]
        xw5 = xw.view(n, wd, wh, ww, C).permute(0,4,1,2,3)
        qkv = self.to_qkv(xw5).chunk(3, dim=1)
        q, k = map(lambda t: rearrange(t, 'n (h d) wd wh ww -> n h (wd wh ww) d', h=h), qkv[:2])

        q = q * self.scale
        attn = einsum('n h i d, n h j d -> n h i j', q, k)
        bias = self.relative_bias[self.relative_index].view(wd*wh*ww, wd*wh*ww, h).permute(2,0,1)
        attn = attn + bias.unsqueeze(0)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        return attn

class WindowAggregate3D(nn.Module):
    def __init__(self, dim, window_size=(11,11,11), heads=4, dim_head=32, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = heads * dim_head

        self.to_v = nn.Conv3d(dim, inner_dim, kernel_size=1, bias=False)
        self.project = nn.Conv3d(inner_dim, dim, kernel_size=1, bias=False) if inner_dim != dim else nn.Identity()
        self.dropout = nn.Dropout(dropout)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, attn, fmap):
        # attn: [n, heads, w3, w3]
        # fmap: [B, C, D, H, W]
        B, C, D, H, W = fmap.shape
        wd, wh, ww = self.window_size
        h, hd = self.heads, self.dim_head
        v = self.to_v(fmap)
        v_windows = rearrange(
            v,
            'B (h d) (d1 wd) (h1 wh) (w1 ww) -> (B d1 h1 w1) h (wd wh ww) d',
            h=h, d=hd,
            wd=wd, wh=wh, ww=ww,
            d1=D//wd, h1=H//wh, w1=W//ww
        )  # [n, heads, w3, d]


        out_win = einsum('n h i j, n h j d -> n h i d', attn, v_windows)
        out_win = self.dropout(out_win)
        out = rearrange(
            out_win,
            '(B d1 h1 w1) h (wd wh ww) d -> B (h d) (d1 wd) (h1 wh) (w1 ww)',
            B=B, h=h, d=hd,
            wd=wd, wh=wh, ww=ww,
            d1=D//wd, h1=H//wh, w1=W//ww
        )
        out = self.project(out)

        return fmap + self.gamma * out


class WindowAggregate3D(nn.Module):
    def __init__(self, dim, window_size=(11,11,11), heads=4, dim_head=32, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = heads * dim_head


        self.to_v = nn.Conv3d(dim, inner_dim, kernel_size=1, bias=False)
        self.project = nn.Conv3d(inner_dim, dim, kernel_size=1, bias=False) if inner_dim != dim else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, attn, fmap):
        # attn: [n, heads, w3, w3]
        # fmap: [B, C, D, H, W]
        B, C, D, H, W = fmap.shape
        wd, wh, ww = self.window_size
        h, hd = self.heads, self.dim_head


        v = self.to_v(fmap)
        v_windows = rearrange(
            v,
            'B (h d) (d1 wd) (h1 wh) (w1 ww) -> (B d1 h1 w1) h (wd wh ww) d',
            h=h, d=hd,
            wd=wd, wh=wh, ww=ww,
            d1=D//wd, h1=H//wh, w1=W//ww
        )


        out_win = einsum('n h i j, n h j d -> n h i d', attn, v_windows)
        out_win = self.dropout(out_win)

        out = rearrange(
            out_win,
            '(B d1 h1 w1) h (wd wh ww) d -> B (h d) (d1 wd) (h1 wh) (w1 ww)',
            B=B, h=h, d=hd,
            wd=wd, wh=wh, ww=ww,
            d1=D//wd, h1=H//wh, w1=W//ww
        )
        out = self.project(out)


        return fmap + self.gamma * out