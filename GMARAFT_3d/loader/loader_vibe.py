__author__ = "Semih Tarik Uenal"

import os
import torch
import numpy as np
from torch.utils.data import Dataset
from random import shuffle
from preprocessing import MetaImageIO  # Adjust this to your actual import


class VibeDatasetPairwise(Dataset):
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
        self.undersampled_file_list = []
        self.fully_file_list = []
        self.undersampling_list = config['undersampling_list']
        self.data_dir = config['data_dir']
        self.filename = "reconstructed.npy"
        self.data_amount = config.get('data_amount', 25000 if mode == 'train' else 28)
        
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
            self.fully_file_list.append(full_path)
            self.undersampled_file_list.append(full_path)
            n = self.fill_list_train(n, n_slices, n_frames, index_us, index_fully)
            index_us += 1
            index_fully += 1
            for us in self.undersampling_list:
                us_folder = f"undersampling_{int(us * 100)}"
                undersampled_path = os.path.join(sample_dir, us_folder, self.filename)
                if not os.path.exists(undersampled_path):
                    continue

                self.fully_file_list.append(full_path)
                self.undersampled_file_list.append(undersampled_path)
                n = self.fill_list_train(n, n_slices, n_frames, index_us, index_fully)
                index_us += 1
                index_fully += 1


    def fill_list_train(self, n, n_slices, n_frames, index_us, index_fully):
        start = int(0.2 * n_slices)
        end = int(0.8 * n_slices)
        for z in range(start, end):
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
                    self.list_info.append(dict)
                    n += 1
        return n
    


    def __getitem__(self, idx):
        entry = self.list_info[idx]
        undersampled = np.load(self.undersampled_file_list[entry['undersampled_idx']])
        fully = np.load(self.fully_file_list[entry['fully_idx']])

        undersampled = np.abs(undersampled)
        fully = np.abs(fully)

        z, t1, t2 = entry['z'], entry['t1'], entry['t2']
        idx1, idx2 = self.get_neighboring_frames(undersampled.shape[1], t2)

        ref_fully = self.normalize(fully[z, t1])
        mov_fully = self.normalize(fully[z, t2])
        context_fully = np.stack([
            self.normalize(fully[z, idx1]),
            self.normalize(fully[z, t2]),
            self.normalize(fully[z, idx2])
        ], axis=0)

        ref_fully = torch.from_numpy(ref_fully[None]).float()
        mov_fully = torch.from_numpy(mov_fully[None]).float()
        context_fully = torch.from_numpy(context_fully).float()

        ref = self.normalize(undersampled[z, t1])
        mov = self.normalize(undersampled[z, t2])
        context = np.stack([
            self.normalize(undersampled[z, idx1]),
            self.normalize(undersampled[z, t2]),
            self.normalize(undersampled[z, idx2])
        ], axis=0)

        ref = torch.from_numpy(ref[None]).float()
        mov = torch.from_numpy(mov[None]).float()
        context = torch.from_numpy(context).float()


        ref = self.center_crop(ref)
        mov = self.center_crop(mov)
        context = self.center_crop(context)
        ref_fully = self.center_crop(ref_fully)
        mov_fully = self.center_crop(mov_fully)
        context_fully = self.center_crop(context_fully)

        return (ref, mov, context), (ref_fully, mov_fully, context_fully)

    def normalize(self, img):
        img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
        return img.astype(np.float32)

    def get_neighboring_frames(self, n_frames, t):
        idx1, idx2 = t - 1, t + 1
        if t == 0:
            idx1 = 11
        if t == (n_frames - 1):
            idx2 = 0
        return idx1, idx2

    def __rmul__(self, v):
        self.list_info = v * self.list_info
        return self
    
    def __len__(self):
        return len(self.list_info)
    
    def center_crop(self, img, crop_h=48, crop_w=224):
        h, w = img.shape[-2:]
        startx = w // 2 - crop_w // 2
        starty = h // 2 - crop_h // 2
        return img[..., starty:starty+crop_h, startx:startx+crop_w]