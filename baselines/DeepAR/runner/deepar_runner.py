import os
import json
from typing import Dict, Optional

import torch
import numpy as np
from tqdm import tqdm
from easytorch.utils.dist import master_only

from basicts.runners import SimpleTimeSeriesForecastingRunner


class DeepARRunner(SimpleTimeSeriesForecastingRunner):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.output_seq_len = cfg["DATASET"]["PARAM"]["output_len"]

    def postprocessing(self, input_data: Dict) -> Dict:
        """Postprocess data.

        Args:
            input_data (Dict): Dictionary containing data to be processed.

        Returns:
            Dict: Processed data.
        """

        if self.scaler is not None and self.scaler.rescale:
            input_data['prediction'] = self.scaler.inverse_transform(input_data['prediction'])
            input_data['target'] = self.scaler.inverse_transform(input_data['target'])
            input_data['inputs'] = self.scaler.inverse_transform(input_data['inputs'])
            if "mus" in input_data.keys():
                input_data['mus'] = self.scaler.inverse_transform(input_data['mus'])
            if "sigmas" in input_data.keys():
                input_data['sigmas'] = self.scaler.inverse_transform(input_data['sigmas'])
        # TODO: add more postprocessing steps as needed.
        return input_data

    def forward(self, data: tuple, epoch:int = None, iter_num: int = None, train:bool = True, **kwargs) -> tuple:
        """feed forward process for train, val, and test. Note that the outputs are NOT re-scaled.

        Args:
            data (tuple): data (future data, history data). [B, L, N, C] for each of them
            epoch (int, optional): epoch number. Defaults to None.
            iter_num (int, optional): iteration number. Defaults to None.
            train (bool, optional): if in the training process. Defaults to True.

        Returns:
            dict: keys that must be included: inputs, prediction, target
        """

        data = self.preprocessing(data)

        # Preprocess input data
        future_data, history_data = data['target'], data['inputs']
        history_data = self.to_running_device(history_data)  # Shape: [B, L, N, C]
        future_data = self.to_running_device(future_data)    # Shape: [B, L, N, C]

        # Select input features
        history_data = self.select_input_features(history_data)
        future_data_4_dec = self.select_input_features(future_data)

        # Forward pass through the model
        model_return = self.model(history_data=history_data, future_data=future_data_4_dec, 
                                  batch_seen=iter_num, epoch=epoch, train=train)

        # parse model return
        if isinstance(model_return, torch.Tensor): model_return = {"prediction": model_return}
        model_return["inputs"] = self.select_target_features(history_data)
        if "target" not in model_return:
            model_return["target"] = self.select_target_features(future_data)
        model_return = self.postprocessing(model_return)
        return model_return
