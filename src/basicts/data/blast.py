import os
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from basicts.utils.constants import BasicTSMode

from .base_dataset import BasicTSDataset


@dataclass
class BLAST(BasicTSDataset):

    """
    BLAST dataset for time series foundation model pretraining.

    Args:
        input_len (int): input length.
        output_len (int, optional): output length. Defaults to None for autoregressive architecture.
            It can be specified for seq2seq architecture.
        mode (BasicTSMode): mode of the dataset.
        local (bool, optional): whether to use local data. Defaults to True.
        data_file_path (str, optional): path to the data file. Defaults to None.
        num_val_samples (int, optional): number of valid samples. Defaults to 1000.
        mixup (bool, optional): whether to use mixup. Defaults to True.
            Mixup augmentation should be consider carefully for seq2seq architecture. Because a seq2seq model 
            must forecast multiple future steps at once, Mixup can introduce abrupt jumps and outliers that 
            the model cannot reliably extrapolate. Lacking sufficient context for long-horizon prediction, 
            the seq2seq architecture ends up wasting capacity on these artifacts. In contrast, an autoregressive
            model is far less affected. Since it predicts just the next time step, it retains ample contextual
            information from the observed history and can learn to forecast accurately.
        k_max (int, optional): k_max for mixup. Defaults to 3.
        alpha (float, optional): alpha for mixup. Defaults to 1.5.
        min_valid_len (int, optional): minimum valid sequence length. Defaults to 1024.
        min_valid_ratio (float, optional): minimum valid ratio. Defaults to 0.5.
        mask_anomaly (bool, optional): whether to mask anomaly. Defaults to False.
        anomaly_threshold (float, optional): anomaly threshold. Defaults to 4.0.

    """

    input_len: int
    output_len: Union[int, None] = None
    mode: Union[BasicTSMode, str] = BasicTSMode.TRAIN
    local: bool = True
    data_file_path: Union[str, None] = None
    num_val_samples: int = 1000
    mixup: bool = True
    k_max: int = 3
    alpha : float = 1.5
    min_valid_len: int = 1024
    min_valid_ratio: float = 0.5
    mask_anomaly: bool = False
    anomaly_threshold: float = 4.0
    dataset_name: str = "BLAST"
    memmap: bool = False

    def __post_init__(self):
        # load data
        self._data = self._load_data()
        self.output_len = self.output_len or 0

        # minimum valid history sequence length
        self.min_seq_length = 48
        # minimum valid future sequence length
        self.min_future_length = 16

    def _load_data(self) -> np.ndarray:

        """
        Load data from local file or remotely.

        Returns:
            np.ndarray: loaded data
        """

        if not self.local:
            # TODO: support download remotely from huggingface
            raise NotImplementedError("Downloading remotely from huggingface is not supported yet.")
        if self.data_file_path is None:
            self.data_file_path = "datasets/BLAST" # default file path
        try:
            shape = np.load(
                os.path.join(self.data_file_path, self.mode, "shape.npy"))
            data = np.memmap(
                os.path.join(
                    self.data_file_path, self.mode, "data.dat"),
                    dtype=np.float32, shape=tuple(shape), mode="r")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Cannot load dataset from {self.data_file_path}, " \
                                    "Please set a correct local path. If you want to download " \
                                    "the dataset, please set the argument `local` to False.") from e

        # use only a subset of the validation set to speed up training
        if self.mode == BasicTSMode.VAL:
            _p = (len(data) - 1) / (self.num_val_samples - 1)
            idx_list = list(range(self.num_val_samples))
            idx_list = [int(_p * i) for i in idx_list]
            data = data[idx_list]

        return data

    def _sample_valid_seq(self):
        """
        Randomly sample a valid sequence from the dataset.

        Returns:
            np.ndarray: a valid sequence
        """
        while True:
            random_idx = np.random.randint(0, self.data.shape[0])
            seq = self.data[random_idx].astype(np.float32)
            valid_len = seq.shape[0] - np.argmax(np.flipud(~np.isnan(seq)))
            valid_point = (~np.isnan(seq)).sum()
            if valid_len < self.min_valid_len:
                continue
            if self.min_seq_length + self.min_future_length > valid_len:
                continue
            if valid_point / valid_len < self.min_valid_ratio:
                continue
            return seq

    def _mixup(self) -> np.ndarray:
        """
        Mixup augmentation.

        Returns:
            np.ndarray: mixed sequence
        """

        # sampling
        k = np.random.randint(1, self.k_max + 1)
        sampled_indices = np.random.choice(len(self), size=(k), replace=True)
        weights = np.random.dirichlet([self.alpha] * k).astype(np.float32)
        seqs = self.data[sampled_indices].astype(np.float32)

        # normalize data
        seqs = np.nan_to_num(seqs, nan=0., posinf=0., neginf=0.)
        seqs = (seqs - np.nanmean(seqs, axis=1, keepdims=True)) / (np.nanstd(seqs, axis=1, keepdims=True) + 1e-8)

        # mixup
        mixup_seq = np.dot(weights, seqs)
        return mixup_seq

    def _get_valid_end_idx(self, seq: np.ndarray, sample_len: int) -> int:
        if not np.isnan(seq[-1]):
            return seq.shape[0] - sample_len
        else:
            last_non_nan_index = seq.shape[0] - np.argmax(np.flipud(~np.isnan(seq)))
            if last_non_nan_index > sample_len:
                return last_non_nan_index - sample_len
            else:
                raise ValueError("No valid end index found in the sequence")

    def _normalize(
            self,
            inputs: np.ndarray,
            targets: Optional[np.ndarray] = None
            ) -> tuple[np.ndarray, np.ndarray]:
        """
        Zscore normalize the inputs and targets.

        Args:
            inputs (np.ndarray): input sequences
            targets (Optional[np.ndarray], optional): target sequences. Defaults to None.

        Returns:
            tuple[np.ndarray, np.ndarray]: normalized inputs and targets
        """

        # zscore normalization
        mean = np.nanmean(inputs)
        std = np.nanstd(inputs)
        if std < 1e-3:
            std = 1
        inputs = (inputs - mean) / std
        if targets is not None:
            targets = (targets - mean) / std
        return inputs, targets

    def _mask_anomaly(self, seq: np.ndarray) -> np.ndarray:
        """
        Mask abnormal values in the sequence.

        Args:
            seq (np.ndarray): input sequence

        Returns:
            np.ndarray: sequence with abnormal values masked
        """

        seq_mask = np.abs(seq) > self.anomaly_threshold
        seq[seq_mask] = np.nan
        return seq

    def _padding_with_nan(self, seq: np.ndarray) -> np.ndarray:
        """
        Pad the sequence with NaN values.

        Args:
            seq (np.ndarray): input sequence

        Returns:
            np.ndarray: padded sequence
        """
        in_pad = self.input_len - self.min_seq_length
        out_pad = self.output_len - self.min_future_length
        seq = np.pad(seq, (in_pad, out_pad), "constant", constant_values=np.nan)
        return seq

    def __getitem__(self, idx: int) -> tuple:

        # sample a valid sequence
        seq = self._mixup() if self.mixup else self._sample_valid_seq()

        # additional padding for seq2seq models
        if self.output_len > 0:
            seq = self._padding_with_nan(seq)

        seq_len = self.input_len + self.output_len
        random_t = np.random.randint(0, self._get_valid_end_idx(seq, seq_len) + 1)
        seq = seq[random_t:random_t + seq_len]

        if self.output_len == 0: # autoregressive models
            # normalize data
            seq, _ = self._normalize(seq)
            # mask abnormal values
            if self.mask_anomaly:
                seq = self._mask_anomaly(seq)
            inputs, targets = seq[:-1], seq[1:]
        else: # seq2seq model
            inputs = seq[:self.input_len]
            targets = seq[self.input_len: self.input_len + self.output_len]
            # normalize data
            inputs, targets = self._normalize(inputs, targets)
            # mask abnormal values
            if self.mask_anomaly:
                inputs = self._mask_anomaly(inputs)
                targets = self._mask_anomaly(targets)

        inputs = np.expand_dims(inputs, axis=-1)
        targets = np.expand_dims(targets, axis=-1)
        return {"inputs": inputs, "targets": targets}

    def __len__(self):
        return self.data.shape[0]

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_data"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._data = self._load_data()

    @property
    def data(self) -> np.ndarray:
        return self._data
