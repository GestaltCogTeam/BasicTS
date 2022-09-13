import pickle

import torch
import numpy as np

from .registry import SCALER_REGISTRY


@SCALER_REGISTRY.register()
def standard_transform(data: np.array, output_dir: str, train_index: list, history_seq_len: int, future_seq_len: int) -> np.array:
    """Standard normalization.

    Args:
        data (np.array): raw time series data.
        output_dir (str): output dir path.
        train_index (list): train index.
        history_seq_len (int): historical sequence length.
        future_seq_len (int): future sequence length.

    Returns:
        np.array: normalized raw time series data.
    """

    # data: L, N, C
    data_train = data[:train_index[-1][1], ...]

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
    data = data * std
    data = data + mean
    return data
