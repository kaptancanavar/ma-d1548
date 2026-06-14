__author__ = "Semih Tarik Uenal"

import os
import torch
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
from preprocessing import MetaImageIO


class VibeDatasetPairwise3D(Dataset):
    def __init__(self, config, mode: str = 'train'):
        self.mode = mode
        self.data_dir = config['data_dir']
        self.filename = "reconstructed.mhd"
        self.target_shape = tuple(config.get('target_shape', (64, 128, 128)))
        self.crop_margin = tuple(config.get('crop_margin', (8, 16, 16)))
        self.morph_iter = int(config.get('morph_iter', 2))
        self.min_foreground_q = float(config.get('min_foreground_q', 0.20))
        self.k_mad = float(config.get('k_mad', 15.0))
        self.undersampling_list = config.get('undersampling_list', [])

        with open(config['data_list'], 'r') as f:
            self.sample_dirs = [os.path.join(self.data_dir, line.strip())
                                for line in f if line.strip()]

        self.list_info = []
        self.fully_data_list = []
        self._bbox_cache = {}
        self._create_lists()
        print(f"[{self.mode}] Loaded {len(self.list_info)} pairs from {len(self.fully_data_list)} cases")

    def _create_lists(self):
        for sample_dir in self.sample_dirs:
            full_path = os.path.join(sample_dir, "fully_sampled", self.filename)
            if not os.path.exists(full_path):
                continue
            vol = MetaImageIO.read(full_path)
            if vol is None or len(vol.shape) != 4:
                continue
            n_frames = vol.shape[0]
            if n_frames != 4:
                continue
            case_idx = len(self.fully_data_list)
            self.fully_data_list.append(full_path)
            for t2 in range(2, n_frames):
                self.list_info.append({
                    'fully_idx': case_idx,
                    't1': 0,
                    't2': t2
                })

    def __len__(self):
        return len(self.list_info)

    def __rmul__(self, v):
        self.list_info = v * self.list_info
        return self

    def __getitem__(self, idx):
        entry = self.list_info[idx]
        fidx = entry['fully_idx']
        t1, t2 = entry['t1'], entry['t2']
        path = self.fully_data_list[fidx]
        vol_np = MetaImageIO.read(path)
        vol_np = self._normalize_per_frame(vol_np)
        vol = torch.from_numpy(vol_np)
        if path not in self._bbox_cache:
            bbox = self._bbox_from_volume_union(vol)
            self._bbox_cache[path] = bbox
        else:
            bbox = self._bbox_cache[path]
        ref = vol[t1][None]
        mov = vol[t2][None]
        ref = self._crop_with_bbox(ref, bbox)
        mov = self._crop_with_bbox(mov, bbox)
        ref = self._resample_to_shape(ref, self.target_shape)
        mov = self._resample_to_shape(mov, self.target_shape)
        return (ref, mov), (ref, mov)

    @staticmethod
    def _normalize_per_frame(vol_np: np.ndarray) -> np.ndarray:
        vol_np = vol_np.astype(np.float32)
        T = vol_np.shape[0]
        vmin = vol_np.reshape(T, -1).min(axis=1)[:, None, None, None]
        vmax = vol_np.reshape(T, -1).max(axis=1)[:, None, None, None]
        norm = 2.0 * (vol_np - vmin) / (vmax - vmin + 1e-8) - 1.0
        return norm

    @staticmethod
    def _normalize01_torch(x: torch.Tensor) -> torch.Tensor:
        x = x - x.min()
        denom = x.max() - x.min()
        return x / (denom + 1e-8)

    def _resample_to_shape(self, tensor_1d: torch.Tensor, target_shape, mode='trilinear') -> torch.Tensor:
        x = tensor_1d.unsqueeze(0)
        out = F.interpolate(x, size=target_shape, mode=mode, align_corners=(mode != 'nearest'))
        return out.squeeze(0)

    @staticmethod
    def _bbox_from_mask(mask_bool: torch.Tensor):
        if not mask_bool.any():
            D, H, W = mask_bool.shape
            return [[0, D], [0, H], [0, W]]
        z, x, y = torch.where(mask_bool)
        return [[int(z.min()), int(z.max()) + 1],
                [int(x.min()), int(x.max()) + 1],
                [int(y.min()), int(y.max()) + 1]]

    def _apply_margin(self, bbox, shape, margin):
        (D, H, W) = shape
        (mz, mx, my) = margin
        (z0, z1), (x0, x1), (y0, y1) = bbox
        z0, z1 = max(0, z0 - mz), min(D, z1 + mz)
        x0, x1 = max(0, x0 - mx), min(H, x1 + mx)
        y0, y1 = max(0, y0 - my), min(W, y1 + my)
        return [[z0, z1], [x0, x1], [y0, y1]]

    def _crop_with_bbox(self, tensor_1d: torch.Tensor, bbox):
        (z0, z1), (x0, x1), (y0, y1) = bbox
        return tensor_1d[:, z0:z1, x0:x1, y0:y1]

    def _bbox_from_volume_union(self, vol: torch.Tensor):
        v = vol.amax(dim=0)
        v = self._normalize01_torch(v)
        v = torch.clamp(v, max=torch.quantile(v, 0.995))
        med = torch.median(v)
        mad = torch.median((v - med).abs()) + 1e-6
        t1 = med + self.k_mad * mad
        t2 = torch.quantile(v, self.min_foreground_q)
        thr = torch.max(t1, t2)
        mask = (v > thr).float().unsqueeze(0).unsqueeze(0)
        for _ in range(self.morph_iter):
            mask = F.max_pool3d(mask, kernel_size=3, stride=1, padding=1)
        for _ in range(self.morph_iter):
            mask = 1.0 - F.max_pool3d(1.0 - mask, kernel_size=3, stride=1, padding=1)
        mask_bool = (mask[0, 0] > 0.5)
        bbox = self._bbox_from_mask(mask_bool)
        bbox = self._apply_margin(bbox, mask_bool.shape, self.crop_margin)
        return bbox
