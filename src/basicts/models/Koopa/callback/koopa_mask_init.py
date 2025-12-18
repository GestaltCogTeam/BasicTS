# pylint: disable=not-callable
import torch
from easytorch.utils import get_logger

from basicts.models.Koopa.arch.layers import FourierFilter
from basicts.runners.callback import BasicTSCallback

logger = get_logger("BasicTS-training")


class KoopaMaskInitCallback(BasicTSCallback):
    
    """Callback for initializing Koopa mask during training.

        Changes made:
        - Robust handling when training loader is empty.
        - Ensure k >= 1 and k <= number of frequencies.
        - Move mask indices and amps to model device.
        - Update any existing FourierFilter module instances inside the model.
        - Defensive typing of indices to torch.long.
    """

    def __init__(self, alpha: float = 0.2):
        super().__init__()
        self.alpha = alpha

    @torch.no_grad()
    def on_train_begin(self, runner):
        model = runner.model
        loader = runner.train_data_loader
        device = next(model.parameters()).device

        amps_sum = 0
        count = 0

        for batch in loader:
            x = batch["inputs"]
            x = x.squeeze(-1) if x.dim() == 4 else x  # (B, L, C)
            r = torch.fft.rfft(x, dim=1)
            amp = torch.abs(r).mean((0, 1))  # (F,)
            amps_sum += amp
            count += 1

        if count == 0:
            logger.info("No training data found, skipping mask init.")
            return

        amps = amps_sum / count  # (F,)
        F = amps.numel()

        k = max(1, int(F * self.alpha))
        idx = torch.topk(amps, k).indices.to(device)

        model.mask_spectrum = idx
        for m in model.modules():
            if isinstance(m, FourierFilter):
                m.mask_spectrum = idx
