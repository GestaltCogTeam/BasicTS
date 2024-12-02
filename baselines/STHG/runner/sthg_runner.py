from typing import Tuple, Union

import torch
import numpy as np
import wandb
import pdb
import os
from basicts.runners import SimpleTimeSeriesForecastingRunner


class STHGRunner(SimpleTimeSeriesForecastingRunner):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
      
    def on_epoch_end(self, epoch: int):
        """Callback at the end of an epoch.

        Args:
            epoch (int): current epoch.
        """
        super().on_epoch_end(epoch)
        self.logger.info('Spatial pct {:.4f}'.format(self.model.w_spatial.mean().item()))

    # def on_test_end(self):
    #     daily_cycle = self.model.dailyQueue.data
    #     weekly_cycle = self.model.weeklyQueue.data
    #     feature_embedding = self.model.node_feature_connection.conv1.weight.squeeze()
    #     w_spatial = self.model.w_spatial
    #     features_all = {'daily_cycle': daily_cycle, 'weekly_cycle': weekly_cycle, 'feature_seq': feature_embedding, 'w_spatial':w_spatial}
    #     test_results = {k: v.cpu().numpy() for k, v in features_all.items()}
    #     np.savez(os.path.join(self.ckpt_save_dir, 'feature_results.npz'), **test_results)


