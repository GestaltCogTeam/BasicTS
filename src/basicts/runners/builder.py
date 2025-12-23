import inspect
from logging import Logger
from typing import TYPE_CHECKING

import torch
from easytorch.device import to_device
from easytorch.utils import get_local_rank, get_world_size
from easytorch.utils.data_prefetcher import DataLoaderX
from easytorch.utils.env import get_rank
from packaging import version
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from basicts.scaler import BasicTSScaler
from basicts.utils import BasicTSMode

if TYPE_CHECKING:
    from basicts.configs import BasicTSConfig


class Builder:

    """
    Builder class for constructing BasicTS objects, including model, dataset, optimizer, lr scheduler, and scaler.
    """

    @staticmethod
    def _build_model(cfg: "BasicTSConfig", logger: Logger) -> torch.nn.Module:
        """Build model.

        Initialize model by calling ```self.define_model```,
        Moves model to the GPU.

        If DDP is initialized, initialize the DDP wrapper.

        Args:
            cfg (Dict): config

        Returns:
            model (nn.Module)
        """

        logger.info("Building model.")
        model = cfg.model(cfg.model_config)
        model = to_device(model)

        # complie model
        if cfg.compile_model:
            # get current torch version
            current_version = torch.__version__
            # torch.compile() is only available in torch>=2.0
            if version.parse(current_version) >= version.parse("2.0"):
                logger.info("Compile model with torch.compile")
                model = torch.compile(model)
            else:
                logger.warning(f"torch.compile requires PyTorch 2.0 or higher. Current version: {current_version}. Skipping compilation.")

        # DDP
        if torch.distributed.is_initialized():
            model = DDP(
                model,
                device_ids=[get_local_rank()],
                find_unused_parameters=cfg.ddp_find_unused_parameters
            )
        return model

    @staticmethod
    def _build_data_loader(cfg: "BasicTSConfig", mode: BasicTSMode, logger: Logger):
        """Build dataloader from BasicTSConfig

        structure of `data_cfg` is
        {
            'BATCH_SIZE': (int, optional) batch size of data loader (default: ``1``),
            'SHUFFLE': (bool, optional) data reshuffled option (default: ``False``),
            'NUM_WORKERS': (int, optional) num workers for data loader (default: ``0``),
            'PIN_MEMORY': (bool, optional) pin_memory option (default: ``False``),
            'PREFETCH': (bool, optional) set to ``True`` to use `DataLoaderX` (default: ``False``),
        }

        Args:
            data_cfg (BasicTSConfig): config
            mode (Literal['train', 'val', 'test']): mode

        Returns:
            data loader
        """

        logger.info(f"Building {mode} data loader.")
        dataset = Builder._build_dataset(cfg, mode)

        sampler = DistributedSampler(
            dataset,
            get_world_size(),
            get_rank(),
            shuffle=cfg.get(f"{mode}_data_shuffle", False)
        ) if torch.distributed.is_initialized() and mode == BasicTSMode.TRAIN else None

        shuffle = False if torch.distributed.is_initialized() and mode == BasicTSMode.TRAIN\
              else cfg.get(f"{mode}_data_shuffle", False)

        return (DataLoaderX if cfg.get(f"{mode}_data_prefetch", False) else DataLoader)(
            dataset,
            collate_fn=cfg.get(f"{mode}_data_collate_fn", None),
            batch_size=cfg.get(f"{mode}_batch_size", 1),
            shuffle=shuffle,
            sampler=sampler,
            num_workers=cfg.get(f"{mode}_data_num_workers", 0),
            pin_memory=cfg.get(f"{mode}_data_pin_memory", False)
        )

    @staticmethod
    def _build_dataset(cfg: "BasicTSConfig", mode: BasicTSMode) -> Dataset:
        """Build a dataset with the given config and mode.

        Args:
            cfg (BasicTSConfig): config
            mode (BasicTSMode): mode

        Returns:
            Dataset: dataset
        """
        if cfg.dataset_params is not None:
            dataset_params = cfg.dataset_params
        else:
            sig = inspect.signature(cfg.dataset_type.__init__)
            dataset_params = {}
            for k, v in sig.parameters.items():
                if k in ["self", "mode"]:
                    continue
                elif k in cfg:
                    dataset_params[k] = cfg[k]
                else:
                    if v.default != inspect.Parameter.empty:
                        dataset_params[k] = v.default
                    else:
                        raise KeyError(f"Key {k} not found in config. Please add it to the config for building the dataset.")
        dataset_params["mode"] = mode
        return cfg.dataset_type(**dataset_params)

    @staticmethod
    def _build_optimizer(cfg: "BasicTSConfig", model: torch.nn.Module) -> Optimizer:
        return cfg.optimizer(model.parameters(), **cfg.optimizer_params)

    @staticmethod
    def _build_lr_scheduler(cfg: "BasicTSConfig", optimizer: torch.optim.Optimizer) -> LRScheduler:
        return cfg.lr_scheduler(optimizer, **cfg.lr_scheduler_params)

    @staticmethod
    def _build_scaler(cfg: "BasicTSConfig") -> BasicTSScaler:
        sig = inspect.signature(cfg.scaler.__init__)
        scaler_params = {}
        for k, v in sig.parameters.items():
            if k == "self":
                continue
            elif k in cfg:
                scaler_params[k] = cfg[k]
            else:
                if v.default != inspect.Parameter.empty:
                    scaler_params[k] = v.default
                else:
                    raise KeyError(f"Key {k} not found in config. Please add it to the config for building the scaler.")
        return cfg.scaler(**scaler_params)
