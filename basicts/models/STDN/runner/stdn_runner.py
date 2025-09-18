import torch
from typing import Dict, Optional
from torch import nn

from basicts.runners import SimpleTimeSeriesForecastingRunner


class STDNRunner(SimpleTimeSeriesForecastingRunner):
    """Runner for DCRNN: add setup_graph and teacher forcing."""

    def __init__(self, cfg: Dict):

        super().__init__(cfg)
        self.lpls = cfg['DATASET']['LPLS']
        self.lpls = self.to_running_device(self.lpls)  # Ensure Laplacian positional encoding is on the correct device
        self.freq = cfg['MODEL']['PARAM']['args']['Data']['time_slice_size']

    def define_model(self, cfg: Dict) -> nn.Module:
        """
        Define the model architecture based on the configuration.

        Args:
            cfg (Dict): Configuration dictionary containing model settings.

        Returns:
            nn.Module: The model architecture.
        """

        model = cfg['MODEL']['ARCH'](**cfg['MODEL']['PARAM'])
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        return model

    def forward(self, data: Dict, epoch: Optional[int] = None, iter_num: Optional[int] = None, train: bool = True, **kwargs) -> Dict:
        """
        Performs the forward pass for training, validation, and testing. 

        Args:
            data (Dict): A dictionary containing 'target' (future data) and 'inputs' (history data) (normalized by self.scaler).
            epoch (int, optional): Current epoch number. Defaults to None.
            iter_num (int, optional): Current iteration number. Defaults to None.
            train (bool, optional): Indicates whether the forward pass is for training. Defaults to True.

        Returns:
            Dict: A dictionary containing the keys:
                  - 'inputs': Selected input features.
                  - 'prediction': Model predictions.
                  - 'target': Selected target features.

        Raises:
            AssertionError: If the shape of the model output does not match [B, L, N].
        """

        data = self.preprocessing(data)

        # Preprocess input data
        future_data, history_data = data['target'], data['inputs']
        history_data = self.to_running_device(history_data)  # Shape: [B, L, N, C]
        future_data = self.to_running_device(future_data)    # Shape: [B, L, N, C]
        batch_size, length, num_nodes, _ = future_data.shape

        # Select input features
        history_data = self.select_input_features(history_data)
        future_data_4_dec = self.select_input_features(future_data)

        if not train:
            # For non-training phases, use only temporal features
            future_data_4_dec[..., 0] = torch.empty_like(future_data_4_dec[..., 0])

        if train:
            mode = 'train'
        else:
            mode = 'test'

        times_all_day = 24*60 / self.freq

        X = history_data[..., [0]]
        TE_h = history_data[..., [2, 1]]
        TE_f = future_data_4_dec[..., [2, 1]]
        #adapt the STND time formate
        TE_h = TE_h * torch.tensor([7, times_all_day], device=TE_h.device).view(1, 1, 2)
        TE_f = TE_f * torch.tensor([7, times_all_day], device=TE_f.device).view(1, 1, 2)
        TE = torch.cat([TE_h, TE_f], dim=1)
        TE = TE[:,:,0,:].squeeze(2)  # Shape: [B, L, 2]
        
        X = self.to_running_device(X)
        TE = self.to_running_device(TE)

        # Forward pass through the model
        model_return = self.model(X, TE, self.lpls, mode)

        # Parse model return
        if isinstance(model_return, torch.Tensor):
            model_return = {'prediction': model_return}
        if 'inputs' not in model_return:
            model_return['inputs'] = self.select_target_features(history_data)
        if 'target' not in model_return:
            model_return['target'] = self.select_target_features(future_data)

        # Ensure the output shape is correct
        assert list(model_return['prediction'].shape)[:3] == [batch_size, length, num_nodes], \
            "The shape of the output is incorrect. Ensure it matches [B, L, N, C]."

        model_return = self.postprocessing(model_return)

        return model_return