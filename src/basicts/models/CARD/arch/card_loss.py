import torch
import torch.nn.functional as F


def h_loss(outputs, targets):
    batch, length, channels = outputs.shape
    levels = [
        level
        for level in (4, 8, 16, 24, 48, 96)
        if level <= length and length % level == 0
    ]
    if not levels:
        return outputs.new_zeros(())

    losses = []
    for level in levels:
        level_outputs = (
            outputs.transpose(-1, -2).reshape(batch, channels, -1, level).mean(dim=-1)
        )
        level_targets = (
            targets.transpose(-1, -2).reshape(batch, channels, -1, level).mean(dim=-1)
        )
        channel_outputs = level_outputs.mean(dim=1, keepdim=True)
        channel_targets = level_targets.mean(dim=1, keepdim=True)
        scale = level**0.5 / 2
        losses.append(
            scale
            * (
                F.l1_loss(level_outputs, level_targets)
                + F.l1_loss(channel_outputs, channel_targets)
            )
        )
    return torch.stack(losses).mean()


def card_loss(prediction, target, pred_len):
    outputs, batch_y = prediction, target

    ratio = (
        torch.arange(1, pred_len + 1, device=prediction.device, dtype=prediction.dtype)
        .rsqrt()
        .unsqueeze(-1)
    )
    outputs = outputs * ratio
    batch_y = batch_y * ratio
    # loss = F.l1_loss(prediction, target)
    # As an auxiliary loss function,
    # it avoids the need for additional primary loss function calculations.

    return h_loss(outputs, batch_y) * 1e-2
