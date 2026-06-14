__author__ = "Semih Tarik Uenal"

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math


class EAE(nn.Module):
    def __init__(self):
        super(EAE, self).__init__()

    def forward(self, inputs, targets):
        EAE_loss = torch.acos((1 + torch.sum(targets * inputs)) /
                              (torch.sqrt(1 + torch.sum(torch.pow(inputs, 2))) *
                               torch.sqrt(1 + torch.sum(torch.pow(targets, 2)))))

        return EAE_loss


class EPE(nn.Module):
    def __init__(self):
        super(EPE, self).__init__()

    def forward(self, inputs, targets):
        EPE_loss = torch.mean(torch.square(targets - inputs))
        return EPE_loss


def gradient(tensor):
    """
    Computes spatial gradients along each axis.
    Supports both 2D (B, C, H, W) and 3D (B, C, D, H, W) tensors.
    Returns a list of first-order differences along spatial dimensions.
    """
    if tensor.dim() == 4:
        dx = tensor[:, :, :, 1:] - tensor[:, :, :, :-1]
        dy = tensor[:, :, 1:, :] - tensor[:, :, :-1, :]
        return [dx, dy]

    elif tensor.dim() == 5:
        dx = tensor[:, :, :, :, 1:] - tensor[:, :, :, :, :-1]
        dy = tensor[:, :, :, 1:, :] - tensor[:, :, :, :-1, :]
        dz = tensor[:, :, 1:, :, :] - tensor[:, :, :-1, :, :]
        return [dx, dy, dz]

    else:
        raise ValueError("Only 2D (4D) and 3D (5D) tensors are supported.")



def smooth_grad_1st(flow, image=None, boundary_awareness=False, alpha=None):
    dx, dy = gradient(flow)
    dx, dy = dx.abs(), dy.abs()
    if boundary_awareness:
        img_dx, img_dy = gradient(image)
        weights_x = torch.exp(-torch.mean(torch.abs(img_dx), 1, keepdim=True) * alpha)
        weights_y = torch.exp(-torch.mean(torch.abs(img_dy), 1, keepdim=True) * alpha)
        loss_x = weights_x * dx / 2.
        loss_y = weights_y * dy / 2.
    else:
        loss_x = dx
        loss_y = dy
    return loss_x.mean() / 2. + loss_y.mean() / 2.

def smooth_grad_1st_3d(flow, image=None, boundary_awareness=False, alpha=None):
    dx, dy, dz = gradient(flow)
    dx, dy, dz = dx.abs(), dy.abs(), dz.abs()

    if boundary_awareness and image is not None:
        img_dx, img_dy, img_dz = gradient(image)
        weights_x = torch.exp(-torch.mean(torch.abs(img_dx), 1, keepdim=True) * alpha)
        weights_y = torch.exp(-torch.mean(torch.abs(img_dy), 1, keepdim=True) * alpha)
        weights_z = torch.exp(-torch.mean(torch.abs(img_dz), 1, keepdim=True) * alpha)
        loss_x = weights_x * dx
        loss_y = weights_y * dy
        loss_z = weights_z * dz
    else:
        loss_x = dx
        loss_y = dy
        loss_z = dz

    return (loss_x.mean() + loss_y.mean() + loss_z.mean()) / 3.0


def smooth_grad_2nd(flo, image=None, boundary_awareness=False, alpha=None):
    dx, dy = gradient(flo)
    dx2, dxdy = gradient(dx)
    dydx, dy2 = gradient(dy)
    eps = 1e-6
    dx2, dy2 = torch.sqrt(dx2 ** 2 + eps), torch.sqrt(dy2 ** 2 + eps)
    if boundary_awareness:
        img_dx, img_dy = gradient(image)
        weights_x = torch.exp(-torch.mean(torch.abs(img_dx), 1, keepdim=True) * alpha)
        weights_y = torch.exp(-torch.mean(torch.abs(img_dy), 1, keepdim=True) * alpha)
        loss_x = weights_x[:, :, :, 1:] * dx2
        loss_y = weights_y[:, :, 1:, :] * dy2
    else:
        loss_x = dx2
        loss_y = dy2
    return loss_x.mean() / 2. + loss_y.mean() / 2.


def smooth_grad_2nd_3d(flow, image=None, boundary_awareness=False, alpha=10):

    dx, dy, dz = gradient(flow)

    dx2, dxdy, dxdz = gradient(dx)
    dydx, dy2, dydz = gradient(dy)
    dzdx, dzdy, dz2 = gradient(dz)

    eps = 1e-6
    dx2 = torch.sqrt(dx2 ** 2 + eps)
    dy2 = torch.sqrt(dy2 ** 2 + eps)
    dz2 = torch.sqrt(dz2 ** 2 + eps)

    if boundary_awareness and image is not None:
        img_dx, img_dy, img_dz = gradient(image)
        weights_x = torch.exp(-torch.mean(torch.abs(img_dx), 1, keepdim=True) * alpha)
        weights_y = torch.exp(-torch.mean(torch.abs(img_dy), 1, keepdim=True) * alpha)
        weights_z = torch.exp(-torch.mean(torch.abs(img_dz), 1, keepdim=True) * alpha)

        loss_x = weights_x[:, :, :, :, 1:] * dx2
        loss_y = weights_y[:, :, :, 1:, :] * dy2
        loss_z = weights_z[:, :, 1:, :, :] * dz2
    else:
        loss_x = dx2
        loss_y = dy2
        loss_z = dz2

    return (loss_x.mean() + loss_y.mean() + loss_z.mean())

# import torch

# def gradient(tensor):
#     """
#     Computes spatial gradients along each axis using torch.gradient.
#     Supports both 2D (B, C, H, W) and 3D (B, C, D, H, W) tensors.
#     Returns a list of first-order differences along spatial dimensions
#     with the same shapes as the original forward-difference implementation:
#       2D: [dx: (B,C,H,W-1), dy: (B,C,H-1,W)]
#       3D: [dx: (B,C,D,H,W-1), dy: (B,C,D,H-1,W), dz: (B,C,D-1,H,W)]
#     """
#     if tensor.dim() == 4:
#         # dims: (B, C, H, W) -> gradients along W (-1) and H (-2)
#         gx, gy = torch.gradient(tensor, dim=(-1, -2), edge_order=1)
#         dx = gx[..., :-1]          # (B,C,H,W-1)
#         dy = gy[:, :, :-1, :]      # (B,C,H-1,W)
#         return [dx, dy]

#     elif tensor.dim() == 5:
#         # dims: (B, C, D, H, W) -> gradients along W (-1), H (-2), D (-3)
#         gx, gy, gz = torch.gradient(tensor, dim=(-1, -2, -3), edge_order=1)
#         dx = gx[..., :-1]               # (B,C,D,H,W-1)
#         dy = gy[:, :, :, :-1, :]        # (B,C,D,H-1,W)
#         dz = gz[:, :, :-1, :, :]        # (B,C,D-1,H,W)
#         return [dx, dy, dz]

#     else:
#         raise ValueError("Only 2D (4D) and 3D (5D) tensors are supported.")


# def smooth_grad_1st(flow, image=None, boundary_awareness=False, alpha=None):
#     dx, dy = gradient(flow)
#     dx, dy = dx.abs(), dy.abs()

#     if boundary_awareness:
#         if image is None or alpha is None:
#             raise ValueError("When boundary_awareness=True, provide image and alpha.")
#         img_dx, img_dy = gradient(image)
#         weights_x = torch.exp(-torch.mean(img_dx.abs(), 1, keepdim=True) * alpha)
#         weights_y = torch.exp(-torch.mean(img_dy.abs(), 1, keepdim=True) * alpha)
#         loss_x = weights_x * dx / 2.
#         loss_y = weights_y * dy / 2.
#     else:
#         loss_x = dx
#         loss_y = dy

#     return loss_x.mean() / 2. + loss_y.mean() / 2.


# def smooth_grad_1st_3d(flow, image=None, boundary_awareness=False, alpha=None):
#     dx, dy, dz = gradient(flow)
#     dx, dy, dz = dx.abs(), dy.abs(), dz.abs()

#     if boundary_awareness:
#         if image is None or alpha is None:
#             raise ValueError("When boundary_awareness=True, provide image and alpha.")
#         img_dx, img_dy, img_dz = gradient(image)
#         weights_x = torch.exp(-torch.mean(img_dx.abs(), 1, keepdim=True) * alpha)
#         weights_y = torch.exp(-torch.mean(img_dy.abs(), 1, keepdim=True) * alpha)
#         weights_z = torch.exp(-torch.mean(img_dz.abs(), 1, keepdim=True) * alpha)
#         loss_x = weights_x * dx / 3.
#         loss_y = weights_y * dy / 3.
#         loss_z = weights_z * dz / 3.
#     else:
#         loss_x = dx
#         loss_y = dy
#         loss_z = dz

#     return (loss_x.mean() + loss_y.mean() + loss_z.mean())


# def smooth_grad_2nd(flo, image=None, boundary_awareness=False, alpha=None):
#     dx, dy = gradient(flo)

#     # Second-order and mixed (not all are used in the loss, but kept for parity)
#     gxx, gxy = torch.gradient(dx, dim=(-1, -2), edge_order=1)  # along W and H
#     gyx, gyy = torch.gradient(dy, dim=(-1, -2), edge_order=1)

#     # Match the original forward-diff shapes: reduce one more element on the diff axis
#     dx2  = torch.sqrt(gxx[..., :-1]**2 + 1e-6)      # (B,C,H,W-2)
#     dy2  = torch.sqrt(gyy[:, :, :-1, :]**2 + 1e-6)  # (B,C,H-2,W)
#     # dxdy = gxy[:, :, :-1, :]        # not used downstream
#     # dydx = gyx[..., :-1]            # not used downstream

#     if boundary_awareness:
#         if image is None or alpha is None:
#             raise ValueError("When boundary_awareness=True, provide image and alpha.")
#         img_dx, img_dy = gradient(image)
#         weights_x = torch.exp(-torch.mean(img_dx.abs(), 1, keepdim=True) * alpha)
#         weights_y = torch.exp(-torch.mean(img_dy.abs(), 1, keepdim=True) * alpha)
#         loss_x = weights_x[:, :, :, 1:] * dx2   # (B,C,H,W-2)
#         loss_y = weights_y[:, :, 1:, :] * dy2   # (B,C,H-2,W)
#     else:
#         loss_x = dx2
#         loss_y = dy2

#     return loss_x.mean() / 2. + loss_y.mean() / 2.


# def smooth_grad_2nd_3d(flow, image=None, boundary_awareness=False, alpha=10):
#     dx, dy, dz = gradient(flow)

#     # Second-order (principal) and mixed terms via torch.gradient
#     gxx, gxy, gxz = torch.gradient(dx, dim=(-1, -2, -3), edge_order=1)
#     gyx, gyy, gyz = torch.gradient(dy, dim=(-1, -2, -3), edge_order=1)
#     gzx, gzy, gzz = torch.gradient(dz, dim=(-1, -2, -3), edge_order=1)

#     # Match original forward-diff shapes: remove one more element on each respective axis
#     eps = 1e-6
#     dx2 = torch.sqrt(gxx[..., :-1]**2 + eps)          # (B,C,D,H,W-2)
#     dy2 = torch.sqrt(gyy[:, :, :, :-1, :]**2 + eps)   # (B,C,D,H-2,W)
#     dz2 = torch.sqrt(gzz[:, :, :-1, :, :]**2 + eps)   # (B,C,D-2,H,W)

#     if boundary_awareness:
#         if image is None or alpha is None:
#             raise ValueError("When boundary_awareness=True, provide image and alpha.")
#         img_dx, img_dy, img_dz = gradient(image)
#         weights_x = torch.exp(-torch.mean(img_dx.abs(), 1, keepdim=True) * alpha)
#         weights_y = torch.exp(-torch.mean(img_dy.abs(), 1, keepdim=True) * alpha)
#         weights_z = torch.exp(-torch.mean(img_dz.abs(), 1, keepdim=True) * alpha)

#         loss_x = weights_x[:, :, :, :, 1:] * dx2   # (B,C,D,H,W-2)
#         loss_y = weights_y[:, :, :, 1:, :] * dy2   # (B,C,D,H-2,W)
#         loss_z = weights_z[:, :, 1:, :, :] * dz2   # (B,C,D-2,H,W)
#     else:
#         loss_x = dx2
#         loss_y = dy2
#         loss_z = dz2

#     return (loss_x.mean() + loss_y.mean() + loss_z.mean())


class Grad2D(nn.Module):
    def __init__(self, mode=1, boundary_awareness=True, alpha=10):
        super(Grad2D, self).__init__()
        self.boundary_awareness = boundary_awareness
        self.alpha = alpha
        assert mode in [1, 2]
        if mode == 2:
            self.func_smooth = smooth_grad_2nd_3d
        elif mode == 1:
            self.func_smooth = smooth_grad_1st_3d

    def forward(self, flow_vec, image):
        return self.func_smooth(flow_vec, image, self.boundary_awareness, self.alpha).mean()
    



class NCCGauss(torch.nn.Module):
    def __init__(self, win=(5,9,9), sigma=(1.5,2.0,2.0), eps=1e-5):
        super().__init__()
        if isinstance(win, int):   win = (win, win, win)
        if isinstance(sigma, (int, float)): sigma = (sigma, sigma, sigma)
        self.win = tuple(int(w) for w in win)
        self.sigma = tuple(float(s) for s in sigma)
        self.eps = eps
        base = self._make_gauss_window(*self.win, *self.sigma)
        self.register_buffer('base_window', base)

    def _make_gauss_window(self, Wd, Wh, Ww, Sd, Sh, Sw):
        zd = torch.arange(Wd, dtype=torch.float32)
        zh = torch.arange(Wh, dtype=torch.float32)
        zw = torch.arange(Ww, dtype=torch.float32)
        gd = torch.exp(-((zd - Wd//2)**2) / (2*Sd**2)); gd /= gd.sum()
        gh = torch.exp(-((zh - Wh//2)**2) / (2*Sh**2)); gh /= gh.sum()
        gw = torch.exp(-((zw - Ww//2)**2) / (2*Sw**2)); gw /= gw.sum()
        G = gd.view(Wd,1,1) * gh.view(1,Wh,1) * gw.view(1,1,Ww)
        return G.unsqueeze(0).unsqueeze(0)  # [1,1,Wd,Wh,Ww]

    def forward(self, y_true, y_pred):
        B, C, D, H, W = y_true.shape
        kd, kh, kw = self.win
        pad = (kw//2, kw//2, kh//2, kh//2, kd//2, kd//2)
        kernel = self.base_window.expand(C,1,kd,kh,kw)

        mu_t = F.conv3d(F.pad(y_true, pad, mode='replicate'), kernel, groups=C)
        mu_p = F.conv3d(F.pad(y_pred, pad, mode='replicate'), kernel, groups=C)

        t0, p0 = y_true - mu_t, y_pred - mu_p
        var_t = F.conv3d(F.pad(t0*t0, pad, mode='replicate'), kernel, groups=C)
        var_p = F.conv3d(F.pad(p0*p0, pad, mode='replicate'), kernel, groups=C)
        cov   = F.conv3d(F.pad(t0*p0, pad, mode='replicate'), kernel, groups=C)

        var_t = torch.clamp(var_t, min=self.eps)
        var_p = torch.clamp(var_p, min=self.eps)

        ncc = cov / torch.sqrt(var_t * var_p)
        # Better: negative correlation (minimize -> maximize NCC)
        return -ncc.mean()
    
class NCC_vxm(torch.nn.Module):
    """
    Local (over window) normalized cross correlation loss

    Adapted from VoxelMorph.
    """
    
    def __init__(self, win=None):
        super(NCC_vxm, self).__init__()
        self.win = win

    def forward(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)
    
class JacobianDeterminant:
    """
    Jacobian determinant metric.
    """
 
    def __init__(self, penalty: str = "l2", loss_mult=1):
        self.loss_mult = loss_mult
        self.penalty = penalty.lower()
        if not self.penalty in ("l1", "l2"):
            raise ValueError("Currently, only L1 and L2 penalty are supported")
 
    def _compute_jacobian(self, y):
        """
        Computes the Jacobian matrix of a displacement field tensor
 
        Args:
            y (torch.Tensor): Displacement field. Must have shape (B, F, *vol_shape, len(vol_shape))
        """
        _y = torch.moveaxis(y, 1, -1)
        vol_shape = _y.shape[1:-1]
        n_dims = len(vol_shape)
        if not len(vol_shape) in (2, 3):
            raise ValueError("Flow has to be 2D or 3D")
 
        # Compute grid
        coords_list = [torch.arange(0, d, device=y.device) for d in vol_shape]
        grid_list = torch.meshgrid(*coords_list, indexing="ij")
        grid = torch.stack(grid_list, dim=-1)
 
        y_total = _y + grid
 
        # Compute Jacobian
        grads = torch.gradient(
            y_total, dim=[d for d in range(1, 1 + n_dims)]
        )  # dx, dy, dz
        J = torch.stack(grads, dim=-1)
 
        return J
 
    def loss(self, _, y_pred):
        J = self._compute_jacobian(y_pred)
        det_J = torch.linalg.det(J)
        if self.penalty == "l1":
            diff = torch.abs(det_J - 1)
        elif self.penalty == "l2":
            diff = torch.square(det_J - 1)
 
        return torch.mean(diff) * self.loss_mult
 
    def forward(self, _, y_pred):
        J = self._compute_jacobian(y_pred)
        det_J = torch.linalg.det(J)
        det_J = det_J.reshape(det_J.shape[0], -1)
        n_non_positive = (det_J <= 0).count_nonzero(dim=-1)
        n_voxels = det_J.shape[-1]
        return n_non_positive / n_voxels
    

class PhotometricLoss(torch.nn.Module):
    def __init__(self, mode='L1'):
        super(PhotometricLoss, self).__init__()
        assert mode in ('L1', 'L2','nvcc')
        self.mode = mode
        if mode == 'L1':
            self.loss = torch.nn.L1Loss(reduction='mean')
        elif mode == 'L2':
            self.loss = torch.nn.MSELoss(reduction='mean')
        elif mode == 'nvcc':
            self.loss = NCC_vxm().cuda()

    def forward(self, inputs, outputs):
        if self.mode == 'L1':
            return (inputs - outputs).abs().mean()

        return self.loss(inputs, outputs)

