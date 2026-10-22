"""Pointwise reconstruction scores and validation-only threshold calibration."""

import json
from pathlib import Path

import numpy as np
import torch
from easytorch.device import set_device_type
from easytorch.utils import set_visible_devices
from torch.utils.data import DataLoader

from basicts.utils import BasicTSMode, RunnerStatus

from .basicts_runner import BasicTSRunner
from .builder import Builder
from .taskflow import BasicTSAnomalyDetectionTaskFlow


@torch.no_grad()
def _score_split(runner, mode):
    dataset = Builder._build_dataset(runner.cfg, mode)
    # Shortcut fields such as cfg.input_len are omitted from serialized configs;
    # the reconstructed dataset retains the authoritative window length.
    length = getattr(dataset, "input_len", None)
    total_points = len(dataset.data)
    if (
        not isinstance(length, int)
        or length <= 0
        or dataset.data.ndim != 2
        or len(dataset) != total_points - length + 1
        or len(dataset) <= 0
    ):
        raise ValueError(
            "Anomaly scoring requires continuous [time, features] data with stride-one windows."
        )
    loader = DataLoader(
        dataset, batch_size=runner.cfg.test_batch_size, shuffle=False, num_workers=0
    )
    sums, counts = (
        np.zeros(total_points, dtype=np.float64),
        np.zeros(total_points, dtype=np.int64),
    )
    offset = 0
    runner.status = RunnerStatus.EVALUATING
    runner.model.eval()
    for step, data in enumerate(loader):
        data = runner.taskflow.preprocess(runner, data)
        output = runner._forward(runner.model, data, step=step)
        output = runner.taskflow.postprocess(runner, output)
        scores = output["anomaly_scores"].cpu().numpy()
        positions = (
            np.arange(offset, offset + scores.shape[0])[:, None]
            + np.arange(length)[None, :]
        )
        valid = np.isfinite(scores)
        np.add.at(sums, positions[valid], scores[valid])
        np.add.at(counts, positions[valid], 1)
        offset += scores.shape[0]
    scores = np.full(total_points, np.nan, dtype=np.float64)
    np.divide(sums, counts, out=scores, where=counts > 0)
    return scores


def anomaly_decisions(calibration_scores, test_scores, labels=None, anomaly_ratio=0.01):
    """Choose a threshold without test data, then compute ordinary pointwise F1.

    Overlapping-window errors must already have been averaged per timestamp.
    Non-finite scores are excluded from calibration and detection metrics.
    """
    calibration_scores = np.asarray(calibration_scores)
    test_scores = np.asarray(test_scores)
    if calibration_scores.ndim != 1 or test_scores.ndim != 1:
        raise ValueError("Anomaly scores must be one-dimensional pointwise arrays.")
    if not 0 < anomaly_ratio < 1:
        raise ValueError("anomaly_ratio must be between 0 and 1.")
    calibration = calibration_scores[np.isfinite(calibration_scores)]
    valid = np.isfinite(test_scores)
    if not calibration.size or not valid.any():
        raise ValueError(
            "Calibration and test splits must contain finite anomaly scores."
        )
    threshold = float(np.quantile(calibration, 1 - anomaly_ratio))
    prediction = valid & (test_scores > threshold)
    report = {
        "threshold": threshold,
        "anomaly_ratio": anomaly_ratio,
        "calibration_points": int(calibration.size),
        "test_points": int(test_scores.size),
        "valid_test_points": int(valid.sum()),
    }
    if labels is not None:
        labels = np.asarray(labels)
        if labels.ndim == 2 and labels.shape[1] == 1:
            labels = labels[:, 0]
        if labels.shape != test_scores.shape or not np.isin(labels, [0, 1]).all():
            raise ValueError("Test labels must contain one binary label per timestamp.")
        truth, detected = labels[valid].astype(bool), prediction[valid]
        tp = int((truth & detected).sum())
        fp = int((~truth & detected).sum())
        fn = int((truth & ~detected).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        report.update(
            {
                "precision": precision,
                "recall": recall,
                "F1": 2
                * precision
                * recall
                / max(precision + recall, np.finfo(float).eps),
            }
        )
    return {
        "metrics": report,
        "scores": test_scores,
        "prediction": prediction,
        "valid_mask": valid,
    }


def evaluate_anomaly_detection(cfg, ckpt_path, output_dir=None):
    """Reload a BasicTS reconstruction checkpoint and score validation/test data.

    Use a normal validation split to calibrate the upper score quantile. The
    scaler is restored from the checkpoint (or fitted on training data only).
    Test labels, when available, are read after scoring and only affect metrics.
    """
    if not isinstance(cfg.taskflow, BasicTSAnomalyDetectionTaskFlow):
        raise ValueError("Use BasicTSAnomalyDetectionConfig for anomaly evaluation.")
    set_device_type("gpu" if cfg.gpus else "cpu")
    if cfg.gpus:
        set_visible_devices(cfg.gpus)
    runner = BasicTSRunner(cfg)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model = getattr(runner.model, "module", runner.model)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if runner.scaler is not None:
        stats = checkpoint.get("scaler_stats")
        if stats:
            runner.scaler.stats = stats
        else:
            runner.scaler.fit(Builder._build_dataset(cfg, BasicTSMode.TRAIN).data)
    calibration = _score_split(runner, BasicTSMode.VAL)
    scores = _score_split(runner, BasicTSMode.TEST)
    labels = None
    labels_path = cfg.anomaly_labels_path
    if labels_path is None:
        data_path = (
            cfg.dataset_params.get("data_file_path") or f"datasets/{cfg.dataset_name}"
        )
        candidate = Path(data_path) / "test_labels.npy"
        if candidate.is_file():
            labels_path = candidate
    if labels_path is not None:
        labels = np.load(labels_path, allow_pickle=False)
    result = anomaly_decisions(calibration, scores, labels, cfg.anomaly_ratio)
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "calibration_scores.npy", calibration)
        np.save(output_dir / "anomaly_scores.npy", result["scores"])
        np.save(output_dir / "prediction.npy", result["prediction"])
        np.save(output_dir / "valid_mask.npy", result["valid_mask"])
        (output_dir / "metrics.json").write_text(
            json.dumps(result["metrics"], indent=2) + "\n"
        )
    return result
