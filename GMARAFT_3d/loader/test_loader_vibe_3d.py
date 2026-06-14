__author__ = "Semih Tarik Uenal"

import os
import torch
import numpy as np
from torch.utils.data import Dataset
from random import shuffle
import torch.nn.functional as F
from preprocessing import MetaImageIO

class VibeDatasetPairwiseTestSet(Dataset):
    def __init__(self, config, mode='train'):
        """
        Args:
            config (dict): Should include:
                - data_list: txt file listing paths to sample directories
                - undersampling_list: e.g. [0.2, 0.33, 0.5]
                - data_amount: max number of samples (optional)
        """
        self.mode = mode
        self.sample_dirs = []
        self.list_info = []
        self.undersampled_data_list = []
        self.fully_data_list = []
        self.undersampling_list = config['undersampling_list']
        self.data_dir = config['data_dir']
        self.filename = "reconstructed.mhd"
        
        with open(config['data_list'], 'r') as f:
            self.sample_dirs = [os.path.join(self.data_dir, line.strip()) for line in f if line.strip()]


        self._create_lists()
        print(f"[{self.mode}] Loaded {len(self.list_info)} samples")

    def _create_lists(self):
        n = 0
        index_us = 0
        index_fully = 0
        for sample_dir in self.sample_dirs:
            full_path = os.path.join(sample_dir, "fully_sampled", self.filename)
            if not os.path.exists(full_path):
                continue

            shape = MetaImageIO.read(full_path).shape  # (slices , frames , lin , par)
            n_frames= shape[0]
            print(n_frames)
            if n_frames!=4:
                continue
            self.fully_data_list.append(full_path)
            self.undersampled_data_list.append(full_path)
            n = self.fill_list_train(n, n_frames, index_us, index_fully)
            # index_us += 1
            # index_fully += 1
            # for us in self.undersampling_list:
            #     us_folder = f"undersampling_{int(us * 100)}"
            #     undersampled_path = os.path.join(sample_dir, us_folder, self.filename)
            #     if not os.path.exists(undersampled_path):
            #         continue

            #     self.fully_data_list.append(np.abs(MetaImageIO.read(full_path)))
            #     self.undersampled_data_list.append(np.abs(MetaImageIO.read(undersampled_path)))
            #     n = self.fill_list_train(n, n_frames, index_us, index_fully)
            #     index_us += 1
            #     index_fully += 1

    def fill_list_train(self, n, n_frames, index_us, index_fully):

        for t1 in range(1):
            for t2 in range(0,n_frames):
                # if n > self.data_amount - 1:
                #     break
                dict = {}
                dict['undersampled_idx'] = index_us
                dict['fully_idx'] = index_fully
                dict['t1'] = t1
                dict['t2'] = t2
                self.list_info.append(dict)
                n += 1
        return n
    


    def __getitem__(self, idx):

        entry = self.list_info[idx]

        # undersampled = self.undersampled_data_list[entry['undersampled_idx']]
        fully = MetaImageIO.read(self.fully_data_list[entry['fully_idx']])

        t1, t2 = entry['t1'], entry['t2']
        # idx1, idx2 = self.get_neighboring_frames(undersampled.shape[0], t2)

        ref_fully = self.normalize(np.abs(fully)[t1])
        mov_fully = self.normalize(np.abs(fully[t2]))
        # context_fully = np.stack([
        #     self.normalize(fully[idx1]),
        #     self.normalize(fully[t2]),
        #     self.normalize(fully[idx2])
        # ], axis=0)

        ref_fully = torch.from_numpy(ref_fully[None]).float()
        mov_fully = torch.from_numpy(mov_fully[None]).float()
        # context_fully = torch.from_numpy(context_fully).float()

        # ref = self.normalize(undersampled[t1])
        # mov = self.normalize(undersampled[t2])
        # context = np.stack([
        #     self.normalize(undersampled[idx1]),
        #     self.normalize(undersampled[t2]),
        #     self.normalize(undersampled[idx2])
        # ], axis=0)

        # ref = torch.from_numpy(ref[None]).float()
        # mov = torch.from_numpy(mov[None]).float()
        # context = torch.from_numpy(context).float()


        # ref = self.center_crop_or_pad(ref)
        # mov = self.center_crop_or_pad(mov)
        # context = self.center_crop_or_pad(context)
        # ref_fully = self.center_crop_or_pad(ref_fully)
        # mov_fully = self.center_crop_or_pad(mov_fully)
        # context_fully = self.center_crop_or_pad(context_fully)
        target_shape = (32, 128, 128)  # Example
        # ref = self.resample_to_shape(ref, target_shape)
        # mov = self.resample_to_shape(mov, target_shape)
        # context = self.resample_to_shape(context, target_shape)
        # ref_fully = self.resample_to_shape(ref_fully, target_shape)
        # mov_fully = self.resample_to_shape(mov_fully, target_shape)
        # context_fully = self.resample_to_shape(context_fully, target_shape)
        return (ref_fully, mov_fully), (ref_fully, mov_fully)
        # return (ref, mov, context), (ref_fully, mov_fully, context_fully)

    def normalize(self, img):
        img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
        return img.astype(np.float32)

    def get_neighboring_frames(self, n_frames, t):
        idx1, idx2 = t - 1, t + 1
        if t == 0:
            idx1 = n_frames-1
        if t == (n_frames - 1):
            idx2 = 0
        return idx1, idx2

    def __rmul__(self, v):
        self.list_info = v * self.list_info
        return self
    
    def __len__(self):
        return len(self.list_info)

    def resample_to_shape(self, tensor, target_shape, mode='trilinear'):
        tensor = tensor.unsqueeze(0)  # (1, C, D, H, W)
        out = F.interpolate(tensor, size=target_shape, mode=mode, align_corners=(mode != 'nearest'))
        return out.squeeze(0)
    
    def center_crop_or_pad(self, img, crop_d=32, crop_h=176, crop_w=176):
        d, h, w = img.shape[-3:]

        pad_d = max(0, crop_d - d)
        pad_h = max(0, crop_h - h)
        pad_w = max(0, crop_w - w)

        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            padding = [ 
                pad_w // 2, pad_w - pad_w // 2,
                pad_h // 2, pad_h - pad_h // 2,
                pad_d // 2, pad_d - pad_d // 2
            ]
            img = F.pad(img, padding, mode='constant', value=0)
            d, h, w = img.shape[-3:]

        startz = d // 2 - crop_d // 2
        starty = h // 2 - crop_h // 2
        startx = w // 2 - crop_w // 2

        return img[..., startz:startz+crop_d, starty:starty+crop_h, startx:startx+crop_w]
    
