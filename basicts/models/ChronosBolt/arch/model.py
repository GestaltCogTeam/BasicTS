import warnings
from typing import Dict, Optional, List

from easydict import EasyDict
import torch
import torch.nn as nn
from transformers import AutoConfig

from .chronos_bolt import ChronosBoltPipeline, ChronosBoltModelForForecasting


class ChronosBolt(nn.Module):
    def __init__(self, model_id: str, from_pretrained: bool, device_map: str):
        # Chronos will load the checkpoint from the model_id
        super().__init__()

        if from_pretrained:
            self.pipeline = ChronosBoltPipeline.from_pretrained(
                model_id,
                trust_remote_code=True,
            )
            self.model = self.pipeline.model
        else:
            config = AutoConfig.from_pretrained(model_id, device_map=device_map)
            self.model = ChronosBoltModelForForecasting(config)
            self.pipeline = ChronosBoltPipeline(model=self.model)

    def forward(self, inputs: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor = None, label_mask: torch.Tensor = None) -> torch.Tensor:
        assert mask is not None and label_mask is not None, "mask and label_mask should not be None"
        context, target = inputs, labels
        mask, target_mask = mask, label_mask
        output = self.model(context=context, target=target, mask=mask, target_mask=target_mask) # ChronosBoltOutput
        loss = output.loss
        return loss

    def generate(self, context: torch.Tensor, prediction_length: int, quantile_levels: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], **prediction_kwargs) -> torch.Tensor:
        # The `predict_quantiles`` function
        _, predictions = self.pipeline.predict_quantiles(
            context=context,
            prediction_length=prediction_length,
            quantile_levels=quantile_levels,
            **prediction_kwargs
        )
        predictions = predictions.to(context.device)

        return predictions
