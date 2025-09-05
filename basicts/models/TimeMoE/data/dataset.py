import pickle

import numpy as np
from torch.utils.data import Dataset


class BLASTDatasetMixUp(Dataset):

    def __init__(self, mode: str, num_valid_samples: int = None, k_max: int = 3, alpha : float = 1.5, postfix: str = None, **kwargs) -> None:
        super().__init__()
        assert mode in ['train', 'valid', 'test', 'val']
        if mode == 'val': mode = 'valid'

        self.mode = mode
        self.alpha = alpha
        self.num_valid_samples = num_valid_samples
        self.k_max = k_max

        if self.mode == 'train':
            if postfix is not None:
                self.mode = f"{self.mode}_{postfix}"
        shape = np.load(f"datasets/BLAST/{self.mode}/shape.npy")
        self.memmap_data = np.memmap(f'datasets/BLAST/{self.mode}/data.dat', dtype=np.float32, shape=tuple(shape), mode='r')

        if self.mode == 'valid' and self.num_valid_samples is not None:
            print(f"Using {self.num_valid_samples} samples for {self.mode} dataset")
            x = self.num_valid_samples
            y = self.memmap_data.shape[0]
            _p = (y - 1) / (x - 1)
            idx_list = list(range(self.num_valid_samples))
            idx_list = [int(_p * i) for i in idx_list]
            self.memmap_data = self.memmap_data[idx_list]

        print(f"Loaded {self.mode} dataset with shape {self.memmap_data.shape}")

    def mixup(self):
        # sampling
        k = np.random.randint(1, self.k_max + 1)
        sampled_indices = np.random.choice(len(self), size=(k), replace=True)
        weights = np.random.dirichlet([self.alpha] * k).astype(np.float32)
        time_series_sampled = self.memmap_data[sampled_indices].astype(np.float32)
        
        # normalize data
        time_series_sampled = np.nan_to_num(time_series_sampled, nan=0., posinf=0., neginf=0.)
        time_series_sampled = (time_series_sampled - np.nanmean(time_series_sampled, axis=1, keepdims=True)) / (np.nanstd(time_series_sampled, axis=1, keepdims=True) + 1e-8)
        
        augmented_batch = np.dot(weights, time_series_sampled)
        return augmented_batch

    def __getitem__(self, idx: int) -> tuple:
        # idx is not used in mixup

        seq = self.mixup()

        # get mask
        mask = (~np.isnan(seq)).astype(np.int32)
        seq = np.nan_to_num(seq, nan=0., posinf=0., neginf=0.)
        # normalize data
        seq = (seq - np.nanmean(seq)) / (np.nanstd(seq) + 1e-8)
        
        # get inputs and labels
        inputs = seq[:-1] # MARK: 需要确保inputs中没有NaN
        labels = seq[1:]
        mask = mask[1:] # MARK: loss mask
        return {'inputs': inputs, 'labels': labels, 'mask': mask}

    def __len__(self):
        return self.memmap_data.shape[0]
