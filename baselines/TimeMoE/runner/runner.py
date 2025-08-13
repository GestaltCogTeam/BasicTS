from typing import Dict

import torch

from basicts.runners.base_utsf_runner import BaseUniversalTimeSeriesForecastingRunner


class TimeMoERunner(BaseUniversalTimeSeriesForecastingRunner):

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        self.context_length = cfg['MODEL']['PARAM']['context_length']

    def forward(self, data: Dict, **kwargs) -> Dict:
        inputs, labels, mask = data['inputs'], data['labels'], data['mask']
        inputs = self.to_running_device(inputs)
        target = self.to_running_device(labels)
        mask = self.to_running_device(mask)
        loss = self.model(context=inputs, target=target, mask=mask) # NOTE: TimeMoE integrates the loss calculation in the forward method
        return {'loss': loss}
