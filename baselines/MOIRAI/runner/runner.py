from typing import Dict

import torch

from basicts.runners import BaseUniversalTimeSeriesForecastingRunner


class MOIRAIEvalRunner(BaseUniversalTimeSeriesForecastingRunner):

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        self.cfg = cfg

    def on_test_start(self, cfg):
        super().on_test_start()
        if 'DATASET' in cfg:
            context_length = cfg['DATASET']['PARAM'].get('input_len', None)
            prediction_length = cfg['DATASET']['PARAM'].get('output_len', None)
        else:
            context_length = cfg['TEST']['DATASET']['PARAM'].get('input_len', None)
            prediction_length = cfg['TEST']['DATASET']['PARAM'].get('output_len', None)
        self.model.update_forecastor(
            context_length=context_length,
            prediction_length=prediction_length
        )

    def inference_forward(self, data: Dict) -> Dict:
        """
        Performs the forward pass for inference. 

        Args:
            data (Dict): A dictionary containing 'target' (future data) and 'inputs' (history data) (normalized by self.scaler).

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

        # For non-training phases, use only temporal features
        future_data_4_dec[..., 0] = torch.empty_like(future_data_4_dec[..., 0])

        # Forward pass through the model
        B, L, N, _ = history_data.shape
        prediction_length = future_data_4_dec.shape[1]
        context = history_data[..., 0].transpose(1, 2).reshape(B * N, L).contiguous()

        # Generate predictions
        model_return = self.model.generate(context=context, **self.generation_params)
        model_return = model_return.reshape(B, N, prediction_length, 1).transpose(1, 2).contiguous()

        # Parse model return
        if isinstance(model_return, torch.Tensor):
            model_return = {'prediction': model_return}
        if 'inputs' not in model_return:
            model_return['inputs'] = self.select_target_features(history_data)
        if 'target' not in model_return:
            model_return['target'] = self.select_target_features(future_data)

        # Ensure the output shape is correct
        assert list(model_return['prediction'].shape)[:3] == [batch_size, length, num_nodes], \
            'The shape of the output is incorrect. Ensure it matches [B, L, N, C].'

        model_return = self.postprocessing(model_return)

        return model_return
