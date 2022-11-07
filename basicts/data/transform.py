import pickle

import torch
import numpy as np

from .registry import SCALER_REGISTRY


@SCALER_REGISTRY.register()
def standard_transform(data: np.array, output_dir: str, train_index: list, history_seq_len: int, future_seq_len: int, heterogeneous: int = False) -> np.array:
    """Standard normalization.

    Args:
        data (np.array): raw time series data.
        output_dir (str): output dir path.
        train_index (list): train index.
        history_seq_len (int): historical sequence length.
        future_seq_len (int): future sequence length.
        heterogeneous (bool): whether the multiple time series is heterogeneous.

    Returns:
        np.array: normalized raw time series data.
    """

    # data: L, N, C, C=1
    data_train = data[:train_index[-1][1], ...]
    if heterogeneous:
        mean, std = data_train.mean(axis=0, keepdims=True), data_train.std(axis=0, keepdims=True)
    else:
        mean, std = data_train[..., 0].mean(), data_train[..., 0].std()

    print("mean (training data):", mean)
    print("std (training data):", std)
    scaler = {}
    scaler["func"] = re_standard_transform.__name__
    scaler["args"] = {"mean": mean, "std": std}
    # label to identify the scaler for different settings.
    with open(output_dir + "/scaler_in{0}_out{1}.pkl".format(history_seq_len, future_seq_len), "wb") as f:
        pickle.dump(scaler, f)

    def normalize(x):
        return (x - mean) / std

    data_norm = normalize(data)
    return data_norm


@SCALER_REGISTRY.register()
def re_standard_transform(data: torch.Tensor, **kwargs) -> torch.Tensor:
    """Standard re-transformation.

    Args:
        data (torch.Tensor): input data.

    Returns:
        torch.Tensor: re-scaled data.
    """

    mean, std = kwargs["mean"], kwargs["std"]
    if isinstance(mean, np.ndarray):
        mean = torch.from_numpy(mean).type_as(data).to(data.device).unsqueeze(0)
        std = torch.from_numpy(std).type_as(data).to(data.device).unsqueeze(0)
    data = data * std
    data = data + mean
    return data


@SCALER_REGISTRY.register()
def min_max_transform(data: np.array, output_dir: str, train_index: list, history_seq_len: int, future_seq_len: int) -> np.array:
    """Min-max normalization.

    Args:
        data (np.array): raw time series data.
        output_dir (str): output dir path.
        train_index (list): train index.
        history_seq_len (int): historical sequence length.
        future_seq_len (int): future sequence length.

    Returns:
        np.array: normalized raw time series data.
    """

    # L, N, C, C=1
    data_train = data[:train_index[-1][1], ...]

    min_value = data_train.min(axis=(0, 1), keepdims=False)[0]
    max_value = data_train.max(axis=(0, 1), keepdims=False)[0]

    print("min: (training data)", min_value)
    print("max: (training data)", max_value)
    scaler = {}
    scaler["func"] = re_min_max_transform.__name__
    scaler["args"] = {"min_value": min_value, "max_value": max_value}
    # label to identify the scaler for different settings.
    # To be fair, only one transformation can be implemented per dataset.
    # TODO: Therefore we (for now) do not distinguish between the data produced by the different transformation methods.
    with open(output_dir + "/scaler_in{0}_out{1}.pkl".format(history_seq_len, future_seq_len), "wb") as f:
        pickle.dump(scaler, f)

    def normalize(x):
        # ref:
        # https://github.com/guoshnBJTU/ASTGNN/blob/f0f8c2f42f76cc3a03ea26f233de5961c79c9037/lib/utils.py#L17
        x = 1. * (x - min_value) / (max_value - min_value)
        x = 2. * x - 1.
        return x

    data_norm = normalize(data)
    return data_norm


@SCALER_REGISTRY.register()
def re_min_max_transform(data: torch.Tensor, **kwargs) -> torch.Tensor:
    """Standard re-min-max transform.

    Args:
        data (torch.Tensor): input data.

    Returns:
        torch.Tensor: re-scaled data.
    """

    min_value, max_value = kwargs["min_value"], kwargs["max_value"]
    # ref:
    # https://github.com/guoshnBJTU/ASTGNN/blob/f0f8c2f42f76cc3a03ea26f233de5961c79c9037/lib/utils.py#L23
    data = (data + 1.) / 2.
    data = 1. * data * (max_value - min_value) + min_value
    return data
