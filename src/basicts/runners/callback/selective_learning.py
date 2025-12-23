from typing import TYPE_CHECKING, Optional

import torch
from easytorch.core.checkpoint import load_ckpt
from easytorch.device import to_device
from easytorch.utils import get_local_rank
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from basicts.configs import BasicTSModelConfig
from basicts.utils import RunnerStatus

from .callback import BasicTSCallback

if TYPE_CHECKING:
    from basicts.runners.basicts_runner import BasicTSRunner


class SelectiveLearning(BasicTSCallback):

    """
    Selective learning callback.
    Paper: Selective Learning for Deep Time Series Forecasting
    Venue: NeurIPS 2025
    Task: Long-term Time Series Forecasting

    Args:
        r_u (float, optional): Uncertainty mask ratio, a float in (0, 1). Default: None.
        r_a (float, optional): Anomaly mask ratio, a float in (0, 1). Default: None.
        estimator (type, optional): Estimation model class for anomaly mask. Default: None.
        estimator_config (BasicTSModelConfig, optional): Config of the estimation model. Default: None.
        ckpt_path (str, optional): Path to the checkpoint of the estimation model. Default: None.
    """

    def __init__(
            self,
            r_u: Optional[float] = None,
            r_a: Optional[float] = None,
            estimator: Optional[type] = None,
            estimator_config: Optional[BasicTSModelConfig] = None,
            ckpt_path: Optional[str] = None):

        super().__init__()

        # config
        self.r_u = r_u
        self.r_a = r_a
        self.estimator = estimator
        self.estimator_config = estimator_config
        self.ckpt_path = ckpt_path

        self.estimation_model = self.estimator(estimator_config)

        if self.r_a is not None and self.estimation_model is None:
            raise RuntimeError("Anomaly mask ratio is set but estimation model is not provided.")
        if self.estimation_model is not None and self.ckpt_path is None:
            raise RuntimeError("Estimation model is set but checkpoint path is not provided.")

        self.history_residual: torch.Tensor = None
        self.num_samples: int = None
        self.uncertainty_mask: torch.Tensor = None

    def on_train_start(self, runner: "BasicTSRunner"):
        runner.logger.info(f"Use selective learning with r_u={self.r_u}, r_a={self.r_a}.")
        self._load_estimator(runner)
        self.estimation_model.eval()
        self.num_samples = len(runner.train_data_loader.dataset)
        runner.train_data_loader = _DataLoaderWithIndex(runner.train_data_loader)

    def on_compute_loss(self, runner: "BasicTSRunner", **kwargs):
        if runner.status == RunnerStatus.TRAINING:
            forward_return = kwargs["forward_return"]
            data = kwargs["data"]
            residual = torch.abs(forward_return["prediction"] - forward_return["targets"])

            # Uncertainty mask
            if self.r_u is not None:
                if self.history_residual is None:
                    _, output_len, num_features = forward_return["targets"].shape
                    self.history_residual = torch.empty(
                        (self.num_samples, output_len, num_features), device="cpu")
                # Update the history residual
                idx: torch.Tensor = data["idx"].to(self.history_residual.device)
                self.history_residual[idx] = residual.cpu()
                expanded_idx = idx.unsqueeze(-1) + torch.arange(runner.cfg["output_len"])
                if self.uncertainty_mask is not None:
                    unc_mask = self.uncertainty_mask[expanded_idx].to(residual.device)
                    forward_return["targets_mask"] = forward_return["targets_mask"] * unc_mask

            # Anomaly mask
            if self.r_a is not None:
                with torch.no_grad():
                    est_foward_return = runner._forward(self.estimation_model, data, step=0)
                residual_lb = torch.abs(est_foward_return["prediction"] - forward_return["targets"])
                dist = residual - residual_lb
                thresholds = torch.quantile(
                    dist, self.r_a, dim=1, keepdim=True)
                ano_mask = dist > thresholds
                forward_return["targets_mask"] = forward_return["targets_mask"] * ano_mask

    def on_epoch_end(self, runner: "BasicTSRunner", **kwargs):
        if self.r_u is not None:
            res_entropy = self._compute_entropy(self.history_residual)
            thresholds = torch.quantile(
                res_entropy, 1 - self.r_u, dim=0, keepdim=True)
            self.uncertainty_mask = res_entropy < thresholds

    def _load_estimator(self, runner: "BasicTSRunner"):

        runner.logger.info(f"Building estimation model {self.estimation_model.__class__.__name__}.")
        self.estimation_model = to_device(self.estimation_model)

        # DDP
        if torch.distributed.is_initialized():
            self.estimation_model = DDP(
                self.estimation_model,
                device_ids=[get_local_rank()],
                find_unused_parameters=runner.cfg.ddp_find_unused_parameters
            )

        # load model weights
        try:
            checkpoint_dict = load_ckpt(None, ckpt_path=self.ckpt_path, logger=runner.logger)
            if isinstance(self.estimation_model, DDP):
                self.estimation_model.module.load_state_dict(checkpoint_dict["model_state_dict"])
            else:
                self.estimation_model.load_state_dict(checkpoint_dict["model_state_dict"])
        except (IndexError, OSError) as e:
            raise OSError(f"Ckpt file {self.ckpt_path} does not exist") from e

    def _compute_entropy(self, residual: torch.Tensor):

        """
        Compute the residual entropy for time series.
        This is an implementation for residual that follows a normal distribution.

        Args:
            residual (torch.Tensor): Residual tensor of shape (N, H, C), where N is the batch size, H is the sequence length, and C is the number of time series.
        Returns:
            torch.Tensor: Residual entropy tensor of shape (N + H - 1, C).
        """

        # tensor shape: N x H x C
        num_samples, output_len, num_features = residual.shape

        # Generate diagonal indices
        ids = (torch.arange(num_samples, device=residual.device)[:, None] + \
            torch.arange(output_len, device=residual.device)[None, :])  # shape (N, H)

        # Flatten and prepare data
        x_flat = residual.view(-1, num_features)  # [N*H, C]
        ids_flat = ids.view(-1, 1).expand(-1, num_features)  # [N*H, C]

        # Initialize result tensors
        result_shape = (num_samples + output_len - 1, num_features)
        sum_per_id = torch.zeros(result_shape, dtype=residual.dtype, device=residual.device)
        sum_squared_per_id = torch.zeros_like(sum_per_id)

        # Compute sum and sum of squares for each id
        sum_per_id.scatter_add_(0, ids_flat, x_flat)
        sum_squared_per_id.scatter_add_(0, ids_flat, (residual ** 2).view(-1, num_features))

        # Compute the number of elements for each id
        counts = torch.bincount(ids.view(-1), minlength=num_samples+output_len-1).to(dtype=residual.dtype)
        counts = counts.unsqueeze(-1).expand(-1, num_features)

        # Compute the residual entropy
        # When the residuals follow a normal distribution, entropy is proportional to variance
        mean = sum_per_id / counts
        var = (sum_squared_per_id / counts) - mean.pow(2)

        return var


class _DataLoaderWithIndex:
    """
    Wrapper for an existing DataLoader. Iteration yields:
      - if original collate returned a dict: that dict with key 'idx' added
      - else: (original_collated_batch, idx_tensor)
    Replace existing dataloader by this wrapper in on_train_start:
      trainer.train_dataloader = DataLoaderWithIndex(trainer.train_dataloader)
    """
    def __init__(self, dataloader: DataLoader):
        self._dataloader = dataloader
        self.dataset = dataloader.dataset
        # use provided collate_fn or fallback to default_collate
        self.collate_fn = dataloader.collate_fn or default_collate
        # use the dataloader's batch_sampler (contains the indices order)
        self.batch_sampler = dataloader.batch_sampler

    def __iter__(self):
        # iterate over batch indices and rebuild batches in main process
        for batch_indices in self.batch_sampler:
            # build the raw sample list exactly like DataLoader would pass to collate_fn
            batch = [self.dataset[i] for i in batch_indices]
            collated = self.collate_fn(batch)  # could be tensor, tuple, list, dict, etc.

            idx_tensor = torch.tensor(batch_indices, dtype=torch.long)  # CPU long by default
            # If collated is a dict, insert 'idx' key for convenience
            if isinstance(collated, dict):
                collated = dict(collated)  # make a shallow copy to be safe
                collated["idx"] = idx_tensor
                yield collated
            else:
                # otherwise keep original structure and append idx as second element
                yield collated, idx_tensor

    def __len__(self):
        # behave like original dataloader
        return len(self._dataloader)

    def __getattr__(self, name):
        # delegate other attributes/methods to underlying dataloader
        return getattr(self._dataloader, name)
