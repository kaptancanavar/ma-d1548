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
output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)

def center_crop_or_pad(img, crop_h=64, crop_w=176):
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

checkpoint_path = os.path.join(cwd, config['trainer']['save_dir'],config['name'], 'checkpoint-epoch5-iter19999.pth')  
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['state_dict'])

print("Model loaded successfully from:", checkpoint_path)




for i, data_list in enumerate(test_loader):
    if i > 0:
        break

    data_blob, target_blob = data_list
    ref, mov, context_img = [x.cuda().squeeze(0) for x in data_blob]
    img_ref_fully, img_mov_fully, context_fully = [x.cuda().squeeze(0) for x in target_blob]

    lin, motion_states, channel, par, col = ref.shape

    z = 220
    t = 7

    img_ref = center_crop_or_pad(ref[z, t])            # (H, W)
    img_mov = center_crop_or_pad(mov[z, t])
    context_img = center_crop_or_pad(context_img[z, t])
    img_mov_fully = center_crop_or_pad(img_mov_fully[z, t])
    img_ref_fully = center_crop_or_pad(img_ref_fully[z, t])

    with torch.no_grad():
        flow_low, flow_pr, context_image_up = model(
            img_ref[None].float().cuda(),
            img_mov[None].float().cuda(),
            context_img[None].float().cuda(),
            test_mode=1
        )
        flow_pr = -flow_pr
        warped_th = warp_torch(img_mov_fully[None].cuda(), flow_pr)
        warped = warped_th[0, 0].cpu().detach().numpy()

    flow = np.transpose(flow_pr[0].cpu().numpy(), (1, 2, 0))  # (H, W, 2)
    flow_img = flow_vis.flow_to_color(flow, convert_to_bgr=False)

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

    axes[0].imshow(img_ref.squeeze().cpu().numpy(), cmap='gray')
    axes[1].imshow(img_mov.squeeze().cpu().numpy(), cmap='gray')
    axes[2].imshow(img_mov_fully.squeeze().cpu().numpy(), cmap='gray')
    add_quiver(axes[2], flow.squeeze(), stride=8, scale=40)
    axes[3].imshow(warped, cmap='gray')
    axes[4].imshow((img_ref - img_mov).squeeze().cpu().numpy(), cmap='RdBu', vmin=-0.5, vmax=0.5)
    axes[5].imshow((img_ref.cpu() - torch.from_numpy(warped).cpu()).squeeze().cpu().numpy(), cmap='RdBu', vmin=-0.5, vmax=0.5)
    diff1 = axes[4].imshow((img_ref - img_mov).squeeze().cpu().numpy(), cmap='RdBu', vmin=-0.5, vmax=0.5)
    diff2 = axes[5].imshow((img_ref.cpu() - torch.from_numpy(warped).cpu()).squeeze().cpu().numpy(), cmap='RdBu', vmin=-0.5, vmax=0.5)

    axes[6].imshow(flow_img.squeeze())
    fig.colorbar(diff1, ax=axes[4], fraction=0.046, pad=0.04)
    fig.colorbar(diff2, ax=axes[5], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()

    ref_np = img_ref.squeeze().cpu().numpy()
    mov_np = img_mov.squeeze().cpu().numpy()
    warped_np = warped if isinstance(warped, np.ndarray) else warped.squeeze().cpu().numpy()

    # Normalize to [0, 255] for saving as PNG
    def normalize_to_uint8(img):
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        return (img * 255).astype(np.uint8)

    imageio.imwrite(os.path.join(output_dir, "ref.png"), normalize_to_uint8(ref_np))
    imageio.imwrite(os.path.join(output_dir, "mov.png"), normalize_to_uint8(mov_np))
    imageio.imwrite(os.path.join(output_dir, "warped.png"), normalize_to_uint8(warped_np))
    photometric_loss = PhotometricLoss()
    ref_f = img_ref_fully[None].float().cuda()
    mov_f = img_mov_fully[None].float().cuda()
    warped_f = warped_th.float().cuda()

    photo_before = photometric_loss(ref_f, mov_f)
    photo_after = photometric_loss(ref_f, warped_f)

    print(f'#########\nPhotometric loss before warping: {photo_before.item():.5f}\n'
          f'Photometric loss after warping: {photo_after.item():.5f}')

