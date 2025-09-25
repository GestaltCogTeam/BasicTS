from typing import Tuple, Union, Dict
import torch
import numpy as np
import wandb
import pdb
import os

from basicts.runners import SimpleTimeSeriesForecastingRunner


class BigSTPreprocessRunner(SimpleTimeSeriesForecastingRunner):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        
        self.tiny_batch_size = cfg.MODEL.PARAM['tiny_batch_size']

    def preprocessing(self, input_data: Dict) -> Dict:
        """Preprocess data.

        Args:
            input_data (Dict): Dictionary containing data to be processed.

        Returns:
            Dict: Processed data.
        """

        input_data = super().preprocessing(input_data)
        
        x = input_data['inputs']
        y = input_data['target']
        
        B, T, N, F = x.shape
        batch_num = int(B * N / self.tiny_batch_size) # 似乎要确保不能等于0
        idx_perm = np.random.permutation([i for i in range(B*N)])

        for j in range(batch_num):
            if j==batch_num-1:
                x_ = x[:, :, idx_perm[(j+1)*self.tiny_batch_size:], :]
                y_ = y[:, :, idx_perm[(j+1)*self.tiny_batch_size:], :]
            else:
                x_ = x[:, :, idx_perm[j*self.tiny_batch_size:(j+1)*self.tiny_batch_size], :]
                y_ = y[:, :, idx_perm[j*self.tiny_batch_size:(j+1)*self.tiny_batch_size], :]

        input_data['inputs'] = x_.transpose(1,2)
        input_data['target'] = y_
        return input_data

   