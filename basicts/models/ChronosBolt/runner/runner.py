from typing import Dict

from basicts.runners.base_utsf_runner import BaseUniversalTimeSeriesForecastingRunner


class ChronosRunner(BaseUniversalTimeSeriesForecastingRunner):

    def __init__(self, cfg: Dict):
        super().__init__(cfg)

    def forward(self, data: Dict, **kwargs) -> Dict:

        inputs, labels, mask, label_mask = data['inputs'], data['labels'], data['mask'], data['target_mask']
        inputs = self.to_running_device(inputs)
        labels = self.to_running_device(labels)
        mask = self.to_running_device(mask)
        label_mask = self.to_running_device(label_mask)
        loss = self.model(inputs=inputs, labels=labels, mask=mask, label_mask=label_mask)
        return {'loss': loss}
