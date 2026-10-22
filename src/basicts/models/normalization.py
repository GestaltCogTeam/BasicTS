"""Configurable normalization/APT for the native BasicTS forecasting API.

Migrated backbones declare their default RevIN statistics. Existing upstream
models opt in through BasicTSForecastingConfig and retain their original default
construction and checkpoint layout.
"""

import copy
import inspect

import torch
from torch import nn

from basicts.metrics import ALL_METRICS
from basicts.modules.norm.forecasting.revin import ForecastRevIN
from basicts.modules.norm.forecasting import APT, DAIN, FAN, SAN, DishTS
from basicts.modules.norm.forecasting.losses import apt_regularization, station_loss


def normalization_enabled(cfg):
    return (
        hasattr(cfg.model, "_input_normalization_defaults")
        or cfg.get("normalization") not in (None, "none", "None")
        or bool(cfg.get("use_apt", False))
    )


class _IdentityNormalization(nn.Module):
    def forward(self, inputs, mode="norm"):
        return inputs


def build_normalized_forecaster(cfg):
    model_type = cfg.model
    supported = {
        "DLinear",
        "NLinear",
        "SparseTSF",
        "Informer",
        "Autoformer",
        "iTransformerForForecasting",
        "PatchTSTForForecasting",
        "CATS",
        "UMixer",
        "BiMamba",
        "CycleNet",
        "DSFormer",
        "Fredformer",
        "ModernTCN",
        "NBeats",
        "NHiTS",
        "OLinear",
        "Pyraformer",
        "SMamba",
        "TQNet",
        "Triformer",
        "WaveNet",
        "CARD",
        "DeepAR",
        "ETSformer",
    }
    if model_type.__name__ not in supported or not model_type.__module__.startswith(
        "basicts.models."
    ):
        raise ValueError(
            f"Normalization/APT has not been adapted for {model_type.__name__}. Supported: {sorted(supported)}"
        )
    model_config = copy.deepcopy(cfg.model_config)
    defaults = dict(getattr(model_type, "_input_normalization_defaults", {}))
    method = cfg.get("normalization", "RevIN" if defaults else None)
    params = {}
    if method == "RevIN":
        params.update(defaults)
        if (
            model_type.__name__ in {"Fredformer", "ModernTCN"}
            and "subtract_last" in model_config
        ):
            params["subtract_last"] = model_config["subtract_last"]
    params.update(cfg.get("normalization_params", {}))
    policy = cfg.get("backbone_normalization", "disable")
    if policy not in {"disable", "keep"}:
        raise ValueError("backbone_normalization must be 'disable' or 'keep'.")
    use_apt = cfg.get("use_apt", False) or cfg.get("normalization") == "APT"
    if (
        use_apt
        and policy == "disable"
        and model_type.__name__ in {"NLinear", "SparseTSF"}
    ):
        raise ValueError(
            f"{model_type.__name__} has fixed input centering that cannot be disabled "
            "through its config and can cancel APT's additive shift. Select another "
            "backbone, or explicitly set backbone_normalization='keep' to retain "
            "that behavior as an experimental combination."
        )
    normalization_flag = "use_norm" if model_type.__name__ == "SMamba" else "use_revin"
    if policy == "disable" and normalization_flag in model_config:
        model_config[normalization_flag] = False
    if model_type.__name__ == "CARD":
        model_config["_external_normalization"] = True
    backbone = model_type(model_config)
    input_len = cfg.input_len or model_config.get("input_len")
    output_len = cfg.output_len or model_config.get("output_len")
    num_features = cfg.get("num_features") or model_config.get("num_features")
    if not all(
        isinstance(n, int) and n > 0 for n in (input_len, output_len, num_features)
    ):
        raise ValueError(
            "Normalization requires positive input_len, output_len and num_features in the config."
        )
    if use_apt and not cfg.dataset_params.get("use_timestamps", True):
        raise ValueError("APT requires use_timestamps=True in the forecasting dataset.")
    wrapped = NormalizedForecaster(
        backbone,
        input_len,
        output_len,
        num_features,
        method,
        params,
        use_apt,
        cfg.get("apt_params", {}),
        cfg.loss,
    )
    if wrapped.apt is not None:
        if cfg.num_steps is not None and wrapped.warmup_epochs:
            raise ValueError(
                "Staged APT training requires epoch-based training (num_steps=None)."
            )
        if cfg.num_steps is None and cfg.num_epochs <= wrapped.warmup_epochs:
            raise ValueError(
                "num_epochs must include at least one joint epoch after APT warmup/calibration."
            )
    # The source APT and DAIN/FAN modules contain inactive parameters. They are
    # preserved for reference checkpoint parity, so DDP must detect unused ones.
    cfg.ddp_find_unused_parameters = True
    return wrapped


class NormalizedForecaster(nn.Module):
    def __init__(
        self,
        backbone,
        input_len,
        output_len,
        num_features,
        normalization=None,
        normalization_params=None,
        use_apt=False,
        apt_params=None,
        loss="MSE",
    ):
        super().__init__()
        self.backbone = backbone
        self.is_probabilistic = type(backbone).__name__ == "DeepAR"
        self.is_card = type(backbone).__name__ == "CARD"
        self.input_len, self.output_len, self.num_features = (
            input_len,
            output_len,
            num_features,
        )
        self.method = {
            None: "none",
            "None": "none",
            "Dish-TS": "DishTS",
            "APT": "none",
        }.get(normalization, normalization)
        use_apt = use_apt or normalization == "APT"
        params = dict(normalization_params or {})
        reserved = {"seq_len", "pred_len", "enc_in"} & params.keys()
        if reserved:
            raise ValueError(
                f"Use the backbone configuration for dimensions, not normalization_params: {reserved}"
            )
        if self.method == "none":
            if params:
                raise ValueError(
                    "normalization_params require an enabled normalization method."
                )
            self.normalization = _IdentityNormalization()
        elif self.method == "RevIN":
            self.normalization = ForecastRevIN(
                num_features=num_features,
                **dict({"eps": 1e-5, "affine": False}, **params),
            )
        elif self.method in {"DishTS", "DAIN", "FAN", "SAN"}:
            if self.method == "SAN":
                period = params.get("period_len", 24)
                if (
                    not isinstance(period, int)
                    or period < 2
                    or input_len % period
                    or output_len % period
                ):
                    raise ValueError(
                        "SAN requires period_len >= 2 dividing input_len and output_len."
                    )
                if (
                    params.get("features", "M") != "M"
                    or params.get("station_type", "adaptive") != "adaptive"
                ):
                    raise ValueError(
                        "The current SAN forecasting pipeline supports features='M', station_type='adaptive'."
                    )
                params.setdefault("period_len", period)
            if self.method == "FAN":
                topk = params.get("freq_topk", 20)
                max_k = (
                    min(input_len, output_len) // 2 + 1
                    if params.get("rfft", True)
                    else min(input_len, output_len)
                )
                if not isinstance(topk, int) or not 1 <= topk <= max_k:
                    raise ValueError(
                        f"FAN freq_topk must be between 1 and {max_k} for these lengths."
                    )
            if self.method == "DAIN" and params.get("mode", "avg") not in {
                None,
                "avg",
                "adaptive_avg",
                "adaptive_scale",
                "full",
            }:
                raise ValueError("Unknown DAIN mode.")
            if self.method == "DishTS" and params.get("init", "avg") not in {
                "standard",
                "avg",
                "uniform",
            }:
                raise ValueError("Unknown DishTS initialization.")
            self.normalization = {
                "DishTS": DishTS,
                "DAIN": DAIN,
                "FAN": FAN,
                "SAN": SAN,
            }[self.method](
                seq_len=input_len, pred_len=output_len, enc_in=num_features, **params
            )
        else:
            raise ValueError(
                f"Unknown normalization {normalization!r}. Use None, RevIN, DishTS, DAIN, FAN or SAN; enable APT with use_apt=True."
            )
        options = dict(apt_params or {})
        if options and not use_apt:
            raise ValueError("apt_params require use_apt=True.")
        self.pretrain_epochs = options.pop("pretrain_epochs", 5) if use_apt else 0
        self.calibration_epochs = (
            options.pop("calibration_epochs", 0 if self.method in {"SAN", "FAN"} else 1)
            if use_apt
            else 0
        )
        if min(self.pretrain_epochs, self.calibration_epochs) < 0:
            raise ValueError("APT epoch counts must be nonnegative.")
        self.warmup_epochs = self.pretrain_epochs + self.calibration_epochs
        self.regularization_weight = options.pop("regularization_weight", 0.25)
        self.scale_epsilon = options.pop("scale_epsilon", 1e-6)
        if self.scale_epsilon <= 0:
            raise ValueError("APT scale_epsilon must be positive.")
        self.apt = APT(num_features=num_features, **options) if use_apt else None
        self.loss_fn = ALL_METRICS[loss] if isinstance(loss, str) else loss
        self.forward_params = set(inspect.signature(backbone.forward).parameters)
        self._required_callbacks = getattr(backbone, "_required_callbacks", [])
        self.normalization_stage = "joint"
        self._original_trainability = {
            name: p.requires_grad for name, p in self.named_parameters()
        }

    def _stage(self, epoch):
        if self.apt is None:
            return "joint"
        if epoch is None and self.warmup_epochs:
            raise ValueError("APT staged training requires the current epoch.")
        epoch = epoch or 1
        if epoch <= self.pretrain_epochs:
            return "apt_pretrain"
        if epoch <= self.warmup_epochs:
            return "apt_calibration"
        return "joint"

    def _base_loss(self, prediction, targets, targets_mask):
        values = {
            "prediction": prediction,
            "targets": targets,
            "targets_mask": targets_mask,
            "input": prediction,
            "target": targets,
        }
        function = (
            self.loss_fn.forward
            if isinstance(self.loss_fn, nn.Module)
            else self.loss_fn
        )
        return self.loss_fn(
            **{
                key: values[key]
                for key in inspect.signature(function).parameters
                if key in values
            }
        )

    def _inverse(self, prediction, weight, bias):
        if self.apt is not None:
            prediction = (prediction - bias) / weight
        return self.normalization(prediction, "denorm")

    def _distribution_affine(self, reference, weight, bias):
        """Future-space affine inverse, including scale for Gaussian likelihoods.

        Every currently supported normalizer has an elementwise affine output
        inverse once its historical statistics have been computed.
        """
        shape = (reference.size(0), self.output_len, self.num_features)
        normalizer = self.normalization
        scale, offset = reference.new_ones(shape), reference.new_zeros(shape)
        if self.method == "RevIN":
            scale = normalizer.stdev
            offset = normalizer.last if normalizer.subtract_last else normalizer.mean
            if normalizer.affine:
                scale = scale / (normalizer.affine_weight + normalizer.eps**2)
                offset = offset - scale * normalizer.affine_bias
        elif self.method == "DishTS":
            scale = (normalizer.xih + 1e-8).sqrt() / normalizer.gamma
            offset = normalizer.phih - scale * normalizer.beta
        elif self.method == "DAIN":
            if normalizer.mode == "avg":
                offset = normalizer.avg.transpose(1, 2)
            elif normalizer.mode is not None:
                offset = normalizer.adaptive_avg.transpose(1, 2)
                if normalizer.mode in {"adaptive_scale", "full"}:
                    scale = normalizer.adaptive_std.transpose(1, 2)
                if normalizer.mode == "full":
                    scale = scale / normalizer.gate.transpose(1, 2)
        elif self.method == "FAN":
            offset = normalizer.pred_main_freq_signal
        elif self.method == "SAN":
            mean, std = normalizer.station_pred.split(self.num_features, dim=-1)
            offset = mean.repeat_interleave(normalizer.period_len, dim=1)
            scale = (std + normalizer.epsilon).repeat_interleave(
                normalizer.period_len, dim=1
            )
        if self.apt is not None:
            scale = scale / weight
            offset = offset - scale * bias
        if (
            not torch.isfinite(scale).all()
            or not torch.isfinite(offset).all()
            or (scale.abs() < 1e-12).any()
        ):
            raise ValueError(
                "DeepAR requires a finite, invertible output normalization scale."
            )
        return scale, offset

    def forward(
        self,
        inputs,
        inputs_timestamps=None,
        targets_timestamps=None,
        targets=None,
        targets_mask=None,
        epoch=None,
        step=None,
        train=None,
        **extra,
    ):
        is_train = self.training if train is None else train
        if inputs.ndim != 3 or inputs.shape[1:] != (self.input_len, self.num_features):
            raise ValueError(
                f"Normalization expects [B, {self.input_len}, {self.num_features}]."
            )
        if is_train:
            self.normalization_stage = self._stage(epoch)
            for prefix, module in (
                ("backbone", self.backbone),
                ("normalization", self.normalization),
            ):
                for name, p in module.named_parameters():
                    p.requires_grad_(
                        self.normalization_stage == "joint"
                        and self._original_trainability[prefix + "." + name]
                    )
                module.train(self.training and self.normalization_stage == "joint")
            if (
                self.is_probabilistic
                and self.training
                and self.normalization_stage == "apt_calibration"
            ):
                # cuDNN RNNs need a training-mode forward to differentiate with
                # respect to inputs, even when all recurrent weights are frozen.
                # DeepAR's LSTM has no recurrent dropout, so values stay unchanged.
                self.backbone.encoder.train()
        normalized = self.normalization(inputs, "norm")
        weight = bias = None
        if self.apt is not None:
            weight, bias = self.apt(
                inputs_timestamps, targets_timestamps, train=is_train
            )
            # Preserve sign while protecting the affine inverse near zero.
            weight = torch.where(
                weight.abs() < self.scale_epsilon,
                torch.where(weight < 0, -self.scale_epsilon, self.scale_epsilon),
                weight,
            )
            normalized = normalized * weight + bias
        kwargs = dict(
            extra,
            inputs_timestamps=inputs_timestamps,
            targets_timestamps=targets_timestamps,
            epoch=epoch,
            step=step,
            train=is_train,
        )
        if type(self.backbone).__name__ == "UMixer":
            kwargs["reference_inputs"] = inputs
        if self.is_card and self.backbone.use_statistic:
            # These tokens describe raw history, independently of the chosen
            # normalization. They must not be recomputed from standardized data.
            kwargs["normalization_stats"] = (
                inputs.mean(dim=1, keepdim=True).detach(),
                (inputs.var(dim=1, keepdim=True, unbiased=False) + 1e-6)
                .sqrt()
                .detach(),
            )
        scale = offset = None
        if self.is_probabilistic:
            scale, offset = self._distribution_affine(inputs, weight, bias)
        if "targets" in self.forward_params:
            # Native seq2seq forecasters require a future decoder tensor.
            # Supply zeros in normalized space, never future observations.
            kwargs["targets"] = normalized.new_zeros(
                inputs.size(0), self.output_len, self.num_features
            )
            if self.is_card or self.is_probabilistic:
                kwargs["targets"] = None
            if self.is_probabilistic and is_train and targets is not None:
                if targets_mask is not None and not targets_mask.bool().all():
                    raise ValueError(
                        "DeepAR teacher forcing currently requires fully observed targets."
                    )
                kwargs["targets"] = (targets - offset) / scale
        # Point forecasters never receive future observations. DeepAR retains
        # training-only teacher forcing; evaluation always autoregresses.
        output = self.backbone(
            normalized, **{k: v for k, v in kwargs.items() if k in self.forward_params}
        )
        result = dict(output) if isinstance(output, dict) else {"prediction": output}
        if self.is_probabilistic:
            # Replace the normalized-space likelihood with the data-space one.
            result.pop("loss", None)
            result["mus"] = self._inverse(result["mus"], weight, bias)
            result["sigmas"] = result["sigmas"] * scale.abs()
        if "loss" in result or "aux_loss" in result:
            raise ValueError(
                "This backbone computes its own loss; a normalization-specific loss adapter is required."
            )
        prediction = result["prediction"]
        if prediction.shape != (inputs.size(0), self.output_len, self.num_features):
            raise ValueError(
                "Normalized forecasting requires matching input/output feature sets."
            )
        result["prediction"] = self._inverse(prediction, weight, bias)
        if is_train and targets is not None:
            if self.normalization_stage == "apt_pretrain":
                result["loss"] = self.regularization_weight * apt_regularization(
                    self.apt, weight, bias
                )
            else:
                if self.is_probabilistic:
                    result["loss"] = (
                        -torch.distributions.Normal(result["mus"], result["sigmas"])
                        .log_prob(targets)
                        .mean()
                    )
                else:
                    result["loss"] = self._base_loss(
                        result["prediction"], targets, targets_mask
                    )
                if self.is_card and self.backbone.use_h_loss:
                    if targets_mask is not None and not targets_mask.bool().all():
                        raise ValueError(
                            "CARD hierarchical loss currently requires fully observed targets."
                        )
                    from basicts.models.CARD.arch.card_loss import card_loss

                    result["aux_loss"] = card_loss(
                        result["prediction"], targets, self.output_len
                    )
                if self.normalization_stage == "joint" and self.method in {
                    "FAN",
                    "SAN",
                }:
                    if targets_mask is not None and not targets_mask.bool().all():
                        raise ValueError(
                            "FAN/SAN auxiliary losses currently require fully observed targets."
                        )
                    auxiliary = (
                        self.normalization.loss(targets)
                        if self.method == "FAN"
                        else station_loss(
                            targets,
                            self.normalization.station_pred,
                            self.normalization.period_len,
                        )
                    )
                    result["normalization_loss"] = auxiliary
                    result["loss"] = result["loss"] + auxiliary
        return result
