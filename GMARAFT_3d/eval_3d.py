__author__ = "Semih Tarik Uenal"

import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from network_3d.model import GMARAFT_Denoiser3D
from train.warp import warp_torch, warp_3d_torch,warp_3d_batch_bspline
from train.losses import PhotometricLoss
from evaluate.utils import add_quiver, increase_brightness, get_data
import flow_vis
from torch.utils.data import DataLoader
from loader.test_loader_vibe_3d import VibeDatasetPairwiseTestSet
import json
import torch.nn.functional as F
import imageio.v2 as imageio
output_dir = "output_images"
from preprocessing import MetaImageIO
os.makedirs(output_dir, exist_ok=True)

import torch
import torch.nn.functional as F
import torch.nn.functional as F

def resample_volume(img, size=(64, 176, 176), mode='trilinear', align_corners=False):

    orig_dim = img.dim()

    if orig_dim == 3:
        img = img.unsqueeze(0).unsqueeze(0)   
    elif orig_dim == 4:
        img = img.unsqueeze(1)                
    
    res = F.interpolate(img, size=size, mode=mode, align_corners=align_corners)

    if orig_dim == 3:
        return res.squeeze(0).squeeze(0)    
    elif orig_dim == 4:
        return res.squeeze(1)                
    return res                              

def center_crop_or_pad(img, crop_d=64, crop_h=176, crop_w=176):
    """
    Applies center crop or symmetric padding to a 3D tensor.
    
    Supports shapes:
    - (D, H, W)
    - (N, D, H, W)
    - (N, C, D, H, W)
    """
    orig_dim = img.dim()

    if orig_dim == 3:
        img = img.unsqueeze(0).unsqueeze(0)  
    elif orig_dim == 4:
        img = img.unsqueeze(1)  
    elif orig_dim == 5:
        pass  
    else:
        raise ValueError(f"Unsupported input dimension: {orig_dim}")

    _, _, d, h, w = img.shape

    pad_d = max(0, crop_d - d)
    pad_h = max(0, crop_h - h)
    pad_w = max(0, crop_w - w)

    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        padding = [pad_w // 2, pad_w - pad_w // 2,
                   pad_h // 2, pad_h - pad_h // 2,
                   pad_d // 2, pad_d - pad_d // 2]
        img = F.pad(img, padding, mode='constant', value=0)

    _, _, d, h, w = img.shape
    startx = w // 2 - crop_w // 2
    starty = h // 2 - crop_h // 2
    startz = d // 2 - crop_d // 2

    img = img[..., startz:startz+crop_d, starty:starty+crop_h, startx:startx+crop_w]

    if orig_dim == 3:
        return img.squeeze(0).squeeze(0)
    elif orig_dim == 4:
        return img.squeeze(1)
    return img


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


model = GMARAFT_Denoiser3D().cuda().eval()

checkpoint_path = os.path.join(cwd, config['trainer']['save_dir'],config['name'], '71_final.pth')  
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['state_dict'])

print("Model loaded successfully from:", checkpoint_path)




for i, data_list in enumerate(test_loader):
    if i!= 3:
        continue

    data_blob, target_blob = data_list
    ref, mov = [x.cuda().squeeze(0) for x in data_blob]

    channel, lin, par, col = ref.shape
    # z = 220
    t = 6

    target_size = (32, 176, 176)  # e.g. (64, 176, 176)
    #target_size = ref.shape[1:]
    ref    = resample_volume(ref,    size=target_size).float().cuda()
    mov    = resample_volume(mov,    size=target_size).float().cuda()


    print(mov.shape)
    with torch.no_grad():
        flow_low, flow_pr = model(
            ref[None].float().cuda(),
            mov[None].float().cuda(),
            test_mode=1
        )
        flow_pr = flow_pr
        print(flow_pr.max())
        print(flow_pr.min())
        warped_th = warp_3d_torch(mov[None].cuda(), flow_pr)
        d_mid = 20
        warped = warped_th[0, 0,d_mid].cpu().detach().numpy()
    
    # Flow for visualization (take center slice)
    flow = np.transpose(flow_pr[-1, :, d_mid].cpu().numpy(), (1, 2, 0))  # [H, W, 3]
    flow_img = flow_vis.flow_to_color(flow[..., 1:3], convert_to_bgr=False)  # only XY
    print(flow.min(),flow.max())
    fig, axes = plt.subplots(1, 7, figsize=(14, 3))
    font = 10
    titles = [
        'I_ref x', 'I_mov x',
        'Moving+flow\n(fully-sampled)',
        'Moving warped\n(fully-sampled)',
        'Moving-Ref',
        'Warped-Ref',
        'Prediction'
    ]

    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=font)
        ax.axis('off')

    axes[0].imshow(ref[0, d_mid].cpu().numpy(), cmap='gray')
    axes[1].imshow(mov[0, d_mid].cpu().numpy(), cmap='gray')
    axes[2].imshow(mov[0, d_mid].cpu().numpy(), cmap='gray')
    add_quiver(axes[2], flow[..., :2], stride=8, scale=40)
    axes[3].imshow(warped, cmap='gray')
    axes[4].imshow((ref[0,  d_mid] - mov[0, d_mid]).cpu().numpy(), cmap='RdBu', vmin=-0.5, vmax=0.5)
    axes[5].imshow((ref[0,  d_mid].cpu() - torch.from_numpy(warped).cpu()).numpy(), cmap='RdBu', vmin=-0.5, vmax=0.5)
    diff1 = axes[4].imshow((ref[0,  d_mid] - mov[0, d_mid]).cpu().numpy(), cmap='RdBu', vmin=-0.5, vmax=0.5)
    diff2 = axes[5].imshow((ref[0,d_mid].cpu() - torch.from_numpy(warped).cpu()).numpy(), cmap='RdBu', vmin=-0.5, vmax=0.5)

    axes[6].imshow(flow_img.squeeze())
    fig.colorbar(diff1, ax=axes[4], fraction=0.046, pad=0.04)
    fig.colorbar(diff2, ax=axes[5], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()

    # Save PNGs
    ref_np = ref[0, 0, d_mid].cpu().numpy()
    mov_np = mov[0, 0, d_mid].cpu().numpy()
    warped_np = warped if isinstance(warped, np.ndarray) else warped.squeeze().cpu().numpy()

    def normalize_to_uint8(img):
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        return (img * 255).astype(np.uint8)
    
    # imageio.imwrite(os.path.join(output_dir, "ref.png"), normalize_to_uint8(ref_np))
    # imageio.imwrite(os.path.join(output_dir, "mov.png"), normalize_to_uint8(mov_np))
    # imageio.imwrite(os.path.join(output_dir, "warped.png"), normalize_to_uint8(warped_np))

    # Photometric loss (use full volumes)
    photometric_loss = PhotometricLoss()
    ref_f = ref[None].float().cuda()
    mov_f = mov[None].float().cuda()
    warped_f = warped_th.float().cuda()

    photo_before = photometric_loss(ref_f, mov_f)
    photo_after = photometric_loss(ref_f, warped_f)

    print(f'#########\nPhotometric loss before warping: {photo_before.item():.5f}\n'
        f'Photometric loss after warping: {photo_after.item():.5f}')
    output_dir = "output_images"
    MetaImageIO.write(
    os.path.join(os.getcwd(),output_dir, "warped_result.mhd"),
    warped_f.squeeze().detach().cpu().numpy()  # (T, Z, H, W)
    )

    # MetaImageIO.write(
    #     os.path.join(os.getcwd(),output_dir, "flow_result.mhd"),
    #     flow.detach().cpu().numpy()  # (T, Z, 2, H, W)
    # )

    MetaImageIO.write(
        os.path.join(os.getcwd(),output_dir, "ref_result.mhd"),
        ref.squeeze().detach().cpu().numpy()
    )

    MetaImageIO.write(
        os.path.join(os.getcwd(),output_dir, "mov_result.mhd"),
        mov.squeeze().detach().cpu().numpy()
    )
