from typing import Dict, Optional

import torch

from basicts.runners.base_tsc_runner import BaseTimeSeriesClassificationRunner


class SimpleTimeSeriesClassificationRunner(BaseTimeSeriesClassificationRunner):
    """
    A Simple Runner for Time Series Classifying: 
    Selects forward and target features. This runner is designed to handle most cases.

    Args:
        cfg (Dict): Configuration dictionary.
    """

    def forward(self, data: Dict, epoch: Optional[int] = None, iter_num: Optional[int] = None, train: bool = True) -> Dict:
        """
        Performs the forward pass for training, validation, and testing. 

        Args:
            data (Dict): A dictionary containing 'target' and 'inputs' (normalized by self.scaler).
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

        # Preprocess input data
        data = self.preprocessing(data)
        target, inputs = data['target'], data['inputs']
        inputs = self.to_running_device(inputs)  # Shape: [B, L, N, C]
        target = self.to_running_device(target)    # Shape: [B, L, N, C]

        # Forward pass through the model
        model_return = self.model(inputs, target, batch_seen=iter_num, epoch=epoch, train=train)

        # Parse model return
        if isinstance(model_return, torch.Tensor):
            model_return = {'prediction': model_return}
        if 'inputs' not in model_return:
            model_return['inputs'] = inputs
        if 'target' not in model_return:
            model_return['target'] = target

        # # Ensure the output shape is correct
        assert list(model_return['prediction'].shape) == [model_return['prediction'].shape[0], self.num_classes], \
             'The shape of the output is incorrect. Ensure it matches [B, num_classes].'

        return model_return

    def preprocessing(self, input_data: Dict) -> Dict:
        """Preprocess data.

        Args:
            input_data (Dict): Dictionary containing data to be processed.

        Returns:
            Dict: Processed data.
        """

        if self.scaler is not None:
            input_data['inputs'] = self.scaler.transform(input_data['inputs'])

        return input_data
