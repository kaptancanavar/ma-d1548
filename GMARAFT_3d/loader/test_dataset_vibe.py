__author__ = "Semih Tarik Uenal"

import os
import torch
import numpy as np
from torch.utils.data import Dataset
from random import shuffle
import torch.nn.functional as F


class VibeDatasetPairwiseTestSet(Dataset):
    def __init__(self, config, mode='test'):
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
        self.undersampled_file_list = []
        self.fully_file_list = []
        self.undersampling_list = config['undersampling_list']
        self.data_dir = config['data_dir']
        self.filename = "reconstructed.npy"
        self.data_amount = 100000
        
        with open(config['data_list'], 'r') as f:
            self.sample_dirs = [os.path.join(self.data_dir, line.strip()) for line in f if line.strip()]

        if self.mode == 'train':
            shuffle(self.sample_dirs)

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

            shape = np.load(full_path, mmap_mode='r').shape  # (slices , frames , lin , par)
            n_slices, n_frames = shape[0], shape[1]
            self.fully_file_list.append(np.abs(np.load(full_path)))
            self.undersampled_file_list.append(np.abs(np.load(full_path)))
            if self.mode =="inference":
                n = self.fill_list_inference(n, n_slices, n_frames, index_us, index_fully)
            else:
                n = self.fill_list_test(n, n_slices, n_frames, index_us, index_fully)
            index_us += 1
            index_fully += 1
            for us in self.undersampling_list:
                us_folder = f"undersampling_{int(us * 100)}"
                undersampled_path = os.path.join(sample_dir, us_folder, self.filename)
                if not os.path.exists(undersampled_path):
                    continue

                self.fully_file_list.append(np.abs(np.load(full_path)))
                self.undersampled_file_list.append(np.abs(np.load(undersampled_path)))
                if self.mode =="inference":
                    n = self.fill_list_inference(n, n_slices, n_frames, index_us, index_fully)
                else:
                    n = self.fill_list_test(n, n_slices, n_frames, index_us, index_fully)
                index_us += 1
                index_fully += 1


    def fill_list_test(self, n, n_slices, n_frames, index_us, index_fully):
        temp_list_pro_dataset = []
        for z in range(n_slices):
            for t1 in range(n_frames):
                for t2 in range(n_frames):
                    if n > self.data_amount - 1:
                        break
                    dict = {}
                    dict['undersampled_idx'] = index_us
                    dict['fully_idx'] = index_fully
                    dict['z'] = z
                    dict['t1'] = t1
                    dict['t2'] = t2
                    temp_list_pro_dataset.append(dict)
                    n += 1
        self.list_info.append(temp_list_pro_dataset)
        return n

    def fill_list_inference(self, n, n_slices, n_frames, index_us, index_fully):
        temp_list_pro_dataset = []
        for z in range(n_slices):
            for t1 in range(1):
                for t2 in range(n_frames):
                    if n > self.data_amount - 1:
                        break
                    dict = {}
                    dict['undersampled_idx'] = index_us
                    dict['fully_idx'] = index_fully
                    dict['z'] = z
                    dict['t1'] = t1
                    dict['t2'] = t2
                    temp_list_pro_dataset.append(dict)
                    n += 1
        self.list_info.append(temp_list_pro_dataset)
        return n

    def __getitem__(self, idx):
        undersampled = self.undersampled_file_list[idx]
        fully = self.fully_file_list[idx]

        Z, T, H, W = undersampled.shape
        t1 = 0  # reference is always t1 = 0

        refs, movs, contexts = [], [], []
        refs_fully, movs_fully, contexts_fully = [], [], []

        for z in range(Z):
            for t2 in range(T):
                idx1, idx2 = self.get_neighboring_frames(T, t2)

                # undersampled
                ref = self.normalize(undersampled[z, t1])[None]  # (1, H, W)
                mov = self.normalize(undersampled[z, t2])[None]
                context = np.stack([
                    self.normalize(undersampled[z, idx1]),
                    self.normalize(undersampled[z, t2]),
                    self.normalize(undersampled[z, idx2])
                ], axis=0)  # (3, H, W)

                ref_f = self.normalize(fully[z, t1])[None]
                mov_f = self.normalize(fully[z, t2])[None]
                context_f = np.stack([
                    self.normalize(fully[z, idx1]),
                    self.normalize(fully[z, t2]),
                    self.normalize(fully[z, idx2])
                ], axis=0)

                refs.append(torch.from_numpy(ref).float())
                movs.append(torch.from_numpy(mov).float())
                contexts.append(torch.from_numpy(context).float())

                refs_fully.append(torch.from_numpy(ref_f).float())
                movs_fully.append(torch.from_numpy(mov_f).float())
                contexts_fully.append(torch.from_numpy(context_f).float())

        refs = torch.stack(refs).view(Z, T, 1, H, W)
        movs = torch.stack(movs).view(Z, T, 1, H, W)
        contexts = torch.stack(contexts).view(Z, T, 3, H, W)

        refs_fully = torch.stack(refs_fully).view(Z, T, 1, H, W)
        movs_fully = torch.stack(movs_fully).view(Z, T, 1, H, W)
        contexts_fully = torch.stack(contexts_fully).view(Z, T, 3, H, W)

        return (refs, movs, contexts), (refs_fully, movs_fully, contexts_fully)



    def normalize(self, img):
        img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
        return img.astype(np.float32)

    def get_neighboring_frames(self, n_frames, t):
        idx1, idx2 = t - 1, t + 1
        if t == 0:
            idx1 = 7
        if t == (n_frames - 1):
            idx2 = 0
        return idx1, idx2

    def __rmul__(self, v):
        self.list_info = v * self.list_info
        return self
    
    def __len__(self):
        return len(self.list_info)


    def center_crop_or_pad(self, img, crop_h=64, crop_w=176):
        h, w = img.shape[-2:]
        
        pad_h = max(0, crop_h - h)
        pad_w = max(0, crop_w - w)

        if pad_h > 0 or pad_w > 0:
            padding = [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            img = F.pad(img, padding, mode='constant', value=0)
            h, w = img.shape[-2:]  

        startx = w // 2 - crop_w // 2
        starty = h // 2 - crop_h // 2
        return img[..., starty:starty+crop_h, startx:startx+crop_w]
