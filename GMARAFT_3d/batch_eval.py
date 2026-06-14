__author__ = "Semih Tarik Uenal"

import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from network.model import GMARAFT_Denoiser
from train.warp import warp_torch
from train.losses import PhotometricLoss
from evaluate.utils import add_quiver, increase_brightness, get_data
import flow_vis
from torch.utils.data import DataLoader
from loader.test_dataset_vibe import VibeDatasetPairwiseTestSet
import json
import torch.nn.functional as F
import imageio.v2 as imageio
from preprocessing import MetaImageIO
import time
import sys 
sys.path.append(
    r"C:\Users\z0043wnf\AppData\Local\miniforge3\Lib\site-packages\bspline_interp-0.0.0-py3.12-win-amd64.egg"
)
import bspline_interp

def pad_image_for_bspline(img: torch.Tensor, padding: int = 2) -> torch.Tensor:
    """
    Pad (Z, Y, X) axes symmetrically for cubic B-spline support.
    """
    return F.pad(img, pad=(padding, padding, padding, padding, padding, padding), mode='reflect')

def warp_3d_batch_bspline(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """
    Warp a 3D complex-valued image using B-spline interpolation and voxel-space motion fields.

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
    M, D, H, W = img.shape  # Unpack dimensions: motion_states, Lin, Par, Col

    img_real = pad_image_for_bspline(img.real, pad)
    img_imag = pad_image_for_bspline(img.imag, pad)

    warped_real = torch.zeros_like(img.real)
    warped_imag = torch.zeros_like(img.imag)

    # for axis in (1, 2, 3):  # axis 1=D, 2=H, 3=W (skip axis 0: motion state)
    #     img_real = bspline_prefilter_1d(img_real, axis)
    #     img_imag = bspline_prefilter_1d(img_imag, axis)
    bspline_interp.bspline_prefilter(img_real.contiguous())
    bspline_interp.bspline_prefilter(img_imag.contiguous())
    z = torch.arange(D, device=img.device)
    y = torch.arange(H, device=img.device)
    x = torch.arange(W, device=img.device)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')
    grid = torch.stack((zz, yy, xx), dim=-1).float().unsqueeze(0).repeat(M, 1, 1, 1, 1)

    coords = grid + flow + pad

    bspline_interp.bspline_interp(img_real.contiguous(), coords.contiguous(), warped_real)
    bspline_interp.bspline_interp(img_imag.contiguous(), coords.contiguous(), warped_imag)
    # warped_real = bspline_interpolate_single(img_real, coords)
    # warped_imag = bspline_interpolate_single(img_imag, coords)
    return warped_real + 1j * warped_imag

def center_crop_or_pad(img, crop_h=64, crop_w=256):
    """
    Supports shapes:
    - (H, W)
    - (N, H, W)
    - (N, C, H, W)
    """
    orig_dim = img.dim()

    if orig_dim == 2:
        img = img.unsqueeze(0).unsqueeze(0) 
    elif orig_dim == 3:
        img = img.unsqueeze(1)  
    elif orig_dim == 4:
        pass  
    else:
        raise ValueError(f"Unsupported input dimension: {orig_dim}")

    _, _, h, w = img.shape
    pad_h = max(0, crop_h - h)
    pad_w = max(0, crop_w - w)

    if pad_h > 0 or pad_w > 0:
        padding = [pad_w // 2, pad_w - pad_w // 2,
                   pad_h // 2, pad_h - pad_h // 2]
        img = F.pad(img, padding, mode='constant', value=0)

    _, _, h, w = img.shape
    startx = w // 2 - crop_w // 2
    starty = h // 2 - crop_h // 2
    img = img[..., starty:starty+crop_h, startx:startx+crop_w]
    if orig_dim == 2:
        return img.squeeze(0).squeeze(0)
    elif orig_dim == 3:
        return img.squeeze(1)
    return img

def reverse_center_crop_or_pad(tensor, target_shape):
    """Restore tensor (C, H, W) to (C, target_H, target_W) using center-padding or cropping."""
    _, h, w = tensor.shape
    target_h, target_w = target_shape

    # Padding
    pad_h = max(target_h - h, 0)
    pad_w = max(target_w - w, 0)
    if pad_h > 0 or pad_w > 0:
        padding = [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]  # left, right, top, bottom
        tensor = torch.nn.functional.pad(tensor, padding)

    # Cropping
    crop_h = max(h - target_h, 0)
    crop_w = max(w - target_w, 0)
    if crop_h > 0 or crop_w > 0:
        crop_top = crop_h // 2
        crop_left = crop_w // 2
        tensor = tensor[:, crop_top:crop_top + target_h, crop_left:crop_left + target_w]

    return tensor


cwd = os.getcwd()
if cwd == '/code':
    cwd = '/z0043wnf/GMRAFT/'

json_file_path = os.path.join(cwd, "configs", "test_vibe_pairwise.json")
with open(json_file_path, 'r') as file:
    config = json.load(file)

config['cwd'] = cwd
config['data_loader']['data_list'] = os.path.join(cwd, config['data_loader']['data_list'])
config['data_loader']['data_dir'] = os.path.join(cwd, config['data_loader']['data_dir'])

test_dataset = VibeDatasetPairwiseTestSet(config['data_loader'], mode='inference')
test_loader = train_loader = DataLoader(test_dataset,
                                   batch_size=config['data_loader']['batch_size'],
                                   pin_memory=True,
                                   shuffle=False,
                                   num_workers=config['data_loader']['num_workers'],
                                   drop_last=True)


model = GMARAFT_Denoiser().cuda().eval()
checkpoint_name = 'checkpoint-epoch5-iter19999'
checkpoint_path = os.path.join(cwd, config['trainer']['save_dir'],config['name'], f'{checkpoint_name}.pth')  
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['state_dict'])

print("Model loaded successfully from:", checkpoint_path)


for i, data_list in enumerate(test_loader):
    if i > 0:
        break  # Only run one batch

    data_blob, target_blob = data_list
    ref, mov, context_img = [x.cuda().squeeze(0) for x in data_blob]
    img_ref_fully, img_mov_fully, context_fully = [x.cuda().squeeze(0) for x in target_blob]

    lin, motion_states, _, par, col = ref.shape
    batch_size = 1

    warped_result = torch.zeros((lin, motion_states, par, col), dtype=torch.float32).cuda()
    flow_result = torch.zeros((lin, motion_states, 2, par, col), dtype=torch.float32).cuda()

    for t in range(motion_states):
        for z_start in range(0, lin, batch_size):
            z_end = min(z_start + batch_size, lin)
            batch_range = range(z_start, z_end)

            # Crop and stack inputs
            ref_batch = torch.stack([
                center_crop_or_pad(ref[z, t]) for z in batch_range
            ]).float().cuda()

            mov_batch = torch.stack([
                center_crop_or_pad(mov[z, t]) for z in batch_range
            ]).float().cuda()

            context_batch = torch.stack([
                center_crop_or_pad(context_img[z, t]) for z in batch_range
            ]).float().cuda()

            mov_fully_batch = torch.stack([
                center_crop_or_pad(img_mov_fully[z, t]) for z in batch_range
            ]).float().cuda()
            start_time = time.time()
            with torch.no_grad():
                flow_low, flow_pr, context_up = model(
                    ref_batch,
                    mov_batch,
                    context_batch,
                    test_mode=1
                )
                flow_pr = -flow_pr

                flow_padded = torch.stack([
                    reverse_center_crop_or_pad(flow_pr[i], (par, col)) for i in range(flow_pr.shape[0])
                ])
                mov_fully_padded = torch.stack([
                    reverse_center_crop_or_pad(mov_fully_batch[i], (par, col)) for i in range(mov_fully_batch.shape[0])
                ])

                warped_batch = warp_torch(mov_fully_padded, flow_padded)
                warped_result[z_start:z_end, t] = warped_batch.squeeze(1)
                flow_result[z_start:z_end, t] = flow_padded
    elapsed_time = time.time() - start_time
    print(f"Inference time for batch {i}, -> {elapsed_time:.3f} seconds")

output_dir =  os.path.join(cwd,"output_images", f'{config['name']}_{checkpoint_name}')
os.makedirs((output_dir), exist_ok=True)

MetaImageIO.write(
    os.path.join(output_dir, "warped_result.mhd"),
    warped_result.permute(1, 0, 2, 3).detach().cpu().numpy()  # (T, Z, H, W)
)

MetaImageIO.write(
    os.path.join(output_dir, "flow_result.mhd"),
    flow_result.permute(1, 0, 2, 3, 4).detach().cpu().numpy()  # (T, Z, 2, H, W)
)

MetaImageIO.write(
    os.path.join(output_dir, "ref.mhd"),
    ref.squeeze(2).permute(1, 0, 2, 3).detach().cpu().numpy()
)

MetaImageIO.write(
    os.path.join(output_dir, "mov.mhd"),
    mov.squeeze(2).permute(1, 0, 2, 3).detach().cpu().numpy()
)

MetaImageIO.write(
    os.path.join(output_dir, "img_ref_fully.mhd"),
    img_ref_fully.squeeze(2).permute(1, 0, 2, 3).detach().cpu().numpy()
)

MetaImageIO.write(
    os.path.join(output_dir, "img_mov_fully.mhd"),
    img_mov_fully.squeeze(2).permute(1, 0, 2, 3).detach().cpu().numpy()
)
