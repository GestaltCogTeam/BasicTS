"""Calendar feature preparation for native forecasting models."""

def calendar_features(timestamps, inputs, length, num_features, center=True):
    """Select known calendar covariates without packing a legacy data tensor.

    BasicTS provides fractions in [0, 1). Some paper implementations consume
    centered covariates; the explicit center argument preserves that convention.
    """
    if timestamps is None:
        result = inputs.new_zeros(inputs.shape[0], length, num_features)
    else:
        if (
            timestamps.ndim != 3
            or timestamps.shape[1] != length
            or timestamps.shape[2] < num_features
        ):
            raise ValueError(
                f"Expected [B, {length}, >= {num_features}] timestamp features."
            )
        result = (
            timestamps[..., :num_features]
            .to(device=inputs.device, dtype=inputs.dtype)
            .clone()
        )
    return (result - 0.5 if center else result).float()
