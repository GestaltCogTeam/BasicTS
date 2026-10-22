"""ETTh1 training recipes for published forecasting models.

Run with PYTHONPATH=src and --model OLinear (the default). Use the standard
BasicTS train/val/test .npy dataset files; optional calendar timestamps must
be prepared for CycleNet and S_Mamba. Model parameters live in each model's
config package. Training settings belong here rather than in model configs.

These recipes use BasicTS validation-based checkpoint selection. Historical
results from other data windowing or test-based selection protocols are not
interchangeable with results from this example.
"""

import argparse
from copy import deepcopy
from importlib import import_module

from torch.optim.lr_scheduler import MultiStepLR

from basicts import BasicTSLauncher
from basicts.configs import BasicTSForecastingConfig
from basicts.runners.callback import EarlyStopping, GradientClipping


# Maximum epochs, batch sizes, and optimizer/scheduler settings are per model.
RECIPES = {
    "BiMamba": {
        "num_epochs": 50,
        "optimizer_params": {"lr": 0.0003, "weight_decay": 0.0},
        "lr_scheduler_params": {"milestones": [1, 25], "gamma": 0.5},
        "train_batch_size": 1,
        "val_batch_size": 1,
        "test_batch_size": 1,
        "patience": 5,
        "clip_grad": 5.0,
        "input_len": 336,
        "num_workers": 4,
    },
    "CARD": {
        "num_epochs": 60,
        "optimizer_params": {"lr": 7.5e-05, "weight_decay": 0.0},
        "lr_scheduler_params": {"milestones": [1, 25], "gamma": 0.5},
        "train_batch_size": 16,
        "val_batch_size": 16,
        "test_batch_size": 16,
        "patience": 5,
        "clip_grad": 5.0,
        "input_len": 336,
        "num_workers": 4,
    },
    "CATS": {
        "num_epochs": 100,
        "optimizer_params": {"lr": 0.0003, "weight_decay": 0.0},
        "lr_scheduler_params": {"milestones": [1, 25], "gamma": 0.5},
        "train_batch_size": 64,
        "val_batch_size": 64,
        "test_batch_size": 64,
        "patience": 5,
        "clip_grad": 5.0,
        "input_len": 336,
        "num_workers": 4,
    },
    "CycleNet": {
        "num_epochs": 100,
        "optimizer_params": {"lr": 0.0003, "weight_decay": 0.0},
        "lr_scheduler_params": {"milestones": [1, 25], "gamma": 0.5},
        "train_batch_size": 64,
        "val_batch_size": 64,
        "test_batch_size": 64,
        "patience": 5,
        "clip_grad": 5.0,
        "input_len": 336,
        "num_workers": 4,
    },
    "DSFormer": {
        "num_epochs": 60,
        "optimizer_params": {"lr": 0.002, "weight_decay": 0.0001},
        "lr_scheduler_params": {"milestones": [1, 5, 15, 25, 50], "gamma": 0.5},
        "train_batch_size": 8,
        "val_batch_size": 8,
        "test_batch_size": 8,
        "patience": 5,
        "clip_grad": 5.0,
        "input_len": 336,
        "num_workers": 4,
    },
    "ETSformer": {
        "num_epochs": 60,
        "optimizer_params": {"lr": 0.0006, "weight_decay": 0.0},
        "lr_scheduler_params": {"milestones": [1, 25], "gamma": 0.5},
        "train_batch_size": 64,
        "val_batch_size": 64,
        "test_batch_size": 64,
        "patience": 5,
        "clip_grad": 5.0,
        "input_len": 336,
        "num_workers": 4,
    },
    "Fredformer": {
        "num_epochs": 60,
        "optimizer_params": {"lr": 0.0006, "weight_decay": 0.0},
        "lr_scheduler_params": {"milestones": [1, 25], "gamma": 0.5},
        "train_batch_size": 1,
        "val_batch_size": 1,
        "test_batch_size": 1,
        "patience": 5,
        "clip_grad": 5.0,
        "input_len": 336,
        "num_workers": 4,
    },
    "NBeats": {
        "num_epochs": 100,
        "optimizer_params": {
            "lr": 0.001,
            "weight_decay": 0,
            "eps": 1e-08,
            "betas": (0.9, 0.95),
        },
        "lr_scheduler_params": {"milestones": [20, 40, 60, 80], "gamma": 0.5},
        "train_batch_size": 32,
        "val_batch_size": 64,
        "test_batch_size": 64,
        "patience": None,
        "clip_grad": 5.0,
        "input_len": 336,
        "num_workers": 0,
    },
    "NHiTS": {
        "num_epochs": 100,
        "optimizer_params": {
            "lr": 0.0005,
            "weight_decay": 0,
            "eps": 1e-08,
            "betas": (0.9, 0.95),
        },
        "lr_scheduler_params": {"milestones": [20, 40, 60, 80], "gamma": 0.5},
        "train_batch_size": 16,
        "val_batch_size": 16,
        "test_batch_size": 16,
        "patience": None,
        "clip_grad": 5.0,
        "input_len": 336,
        "num_workers": 0,
    },
    "OLinear": {
        "num_epochs": 60,
        "optimizer_params": {"lr": 3.75e-05, "weight_decay": 0.0},
        "lr_scheduler_params": {"milestones": [1, 25], "gamma": 0.5},
        "train_batch_size": 64,
        "val_batch_size": 64,
        "test_batch_size": 64,
        "patience": 5,
        "clip_grad": 5.0,
        "input_len": 336,
        "num_workers": 4,
    },
    "Pyraformer": {
        "num_epochs": 100,
        "optimizer_params": {"lr": 0.0002, "weight_decay": 0},
        "lr_scheduler_params": {"milestones": [1, 25], "gamma": 0.5},
        "train_batch_size": 64,
        "val_batch_size": 64,
        "test_batch_size": 64,
        "patience": None,
        "clip_grad": 5.0,
        "input_len": 96,
        "num_workers": 0,
    },
    "S_Mamba": {
        "num_epochs": 60,
        "optimizer_params": {"lr": 9.375e-06, "weight_decay": 0.0},
        "lr_scheduler_params": {"milestones": [1, 25], "gamma": 0.5},
        "train_batch_size": 8,
        "val_batch_size": 8,
        "test_batch_size": 8,
        "patience": 5,
        "clip_grad": 5.0,
        "input_len": 336,
        "num_workers": 4,
    },
    "TQNet": {
        "num_epochs": 60,
        "optimizer_params": {"lr": 7.5e-05, "weight_decay": 0.0},
        "lr_scheduler_params": {"milestones": [1, 25], "gamma": 0.5},
        "train_batch_size": 16,
        "val_batch_size": 16,
        "test_batch_size": 16,
        "patience": 5,
        "clip_grad": 5.0,
        "input_len": 336,
        "num_workers": 4,
    },
    "Triformer": {
        "num_epochs": 100,
        "optimizer_params": {"lr": 0.0002, "weight_decay": 0.0001},
        "lr_scheduler_params": {"milestones": [1, 25], "gamma": 0.5},
        "train_batch_size": 64,
        "val_batch_size": 64,
        "test_batch_size": 64,
        "patience": None,
        "clip_grad": 5.0,
        "input_len": 336,
        "num_workers": 0,
    },
    "UMixer": {
        "num_epochs": 100,
        "optimizer_params": {"lr": 5e-05, "weight_decay": 0},
        "lr_scheduler_params": {"milestones": [1, 5], "gamma": 0.5},
        "train_batch_size": 64,
        "val_batch_size": 64,
        "test_batch_size": 64,
        "patience": None,
        "clip_grad": None,
        "input_len": 720,
        "num_workers": 0,
    },
    "WaveNet": {
        "num_epochs": 25,
        "optimizer_params": {"lr": 0.002, "weight_decay": 0.0001},
        "lr_scheduler_params": {"milestones": [1, 50], "gamma": 0.5},
        "train_batch_size": 16,
        "val_batch_size": 16,
        "test_batch_size": 16,
        "patience": None,
        "clip_grad": 5.0,
        "input_len": 336,
        "num_workers": 0,
    },
}


def build_config(
    model_name="OLinear",
    gpus="0",
    data_file_path=None,
    use_revin=None,
    *,
    normalization="RevIN",
    use_apt=False,
    normalization_params=None,
    apt_params=None,
):
    """Build a training configuration without loading data or starting a run."""
    recipe = deepcopy(RECIPES[model_name])
    length = recipe.pop("input_len")
    patience = recipe.pop("patience")
    clip_grad = recipe.pop("clip_grad")
    workers = recipe.pop("num_workers")
    module = import_module(f"basicts.models.{model_name}")
    class_name = "SMamba" if model_name == "S_Mamba" else model_name
    model_type = getattr(module, class_name)
    config_type = getattr(module, f"{class_name}Config")
    model_config = config_type(input_len=length, output_len=336, num_features=7)
    if use_revin is not None:
        normalization = "RevIN" if use_revin else None
    callbacks = [
        callback() for callback in getattr(model_type, "_required_callbacks", [])
    ]
    if clip_grad is not None:
        callbacks.append(GradientClipping(clip_grad))
    if patience is not None:
        callbacks.append(EarlyStopping(patience))
    dataset_params = {
        "use_timestamps": use_apt
        or normalization == "APT"
        or model_name in {"CycleNet", "S_Mamba"}
    }
    if data_file_path is not None:
        dataset_params["data_file_path"] = data_file_path
    return BasicTSForecastingConfig(
        model=model_type,
        model_config=model_config,
        normalization=normalization,
        normalization_params=normalization_params or {},
        use_apt=use_apt,
        apt_params=apt_params or {},
        dataset_name="ETTh1",
        dataset_params=dataset_params,
        input_len=length,
        output_len=336,
        gpus=gpus,
        seed=42,
        callbacks=callbacks,
        metrics=["MAE", "MSE"],
        loss="MAE",
        target_metric="MAE",
        norm_each_channel=True,
        rescale=False,
        lr_scheduler=MultiStepLR,
        train_data_num_workers=workers,
        val_data_num_workers=workers,
        test_data_num_workers=workers,
        train_data_pin_memory=True,
        val_data_pin_memory=True,
        test_data_pin_memory=True,
        test_interval=1,
        eval_after_train=True,
        eval_horizons=[12, 24, 48, 96, 192, 288, 336],
        tf32=False,
        deterministic=False,
        cudnn_benchmark=True,
        save_results=True,
        **recipe,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(RECIPES), default="OLinear")
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--data-file-path")
    normalization = parser.add_mutually_exclusive_group()
    normalization.add_argument(
        "--normalization",
        choices=["none", "RevIN", "DishTS", "DAIN", "FAN", "SAN", "APT"],
        default="RevIN",
    )
    normalization.add_argument(
        "--use-revin", dest="normalization", action="store_const", const="RevIN"
    )
    normalization.add_argument(
        "--no-revin", dest="normalization", action="store_const", const="none"
    )
    parser.add_argument("--use-apt", action="store_true")
    args = parser.parse_args()
    BasicTSLauncher.launch_training(
        build_config(
            args.model,
            args.gpus,
            args.data_file_path,
            normalization=None if args.normalization == "none" else args.normalization,
            use_apt=args.use_apt,
        )
    )


if __name__ == "__main__":
    main()
