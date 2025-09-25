import random

import torch
from torch.utils.data.distributed import DistributedSampler


class InfiniteGenerator:
    """Infinite data loader."""

    def __init__(self, dataloader) -> None:
        self.dataloader = dataloader
        self.iterator = iter(dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            # set iteration for sampler in distributed mode
            # see https://pytorch.org/docs/stable/data.html
            sampler = self.dataloader.sampler
            if torch.distributed.is_initialized() and isinstance(sampler, DistributedSampler) and sampler.shuffle:
                self.dataloader.sampler.set_epoch(random.randint(0, 1e7))
            self.iterator = iter(self.dataloader)
            return next(self.iterator)
