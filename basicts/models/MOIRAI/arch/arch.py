from typing import Dict

import torch
import torch.nn as nn

from .moirai import MoiraiForecast, MoiraiModule


class MOIRAI(nn.Module):
    "NOTE: only support inference now, no training."
    def __init__(self, prediction_length, context_length, patch_size, model_args: Dict = None, from_pretrained: bool = None):
        assert model_args is not None or from_pretrained is not None, "Either model_args or model_id and from_pretrained must be provided."
        assert patch_size in ["auto", 8, 16, 32, 64, 128]

        self.context_length = context_length
        self.prediction_length = prediction_length
        self.patch_size = patch_size

        if self.patch_size == "auto":
            # MOIRAI use longer context to auto-determine patch size
            print("When automatically determining the patch size, MOIRAI uses the last few points as pseudo-labels, and utilizes the remaining points as context to determine the patch size. As a result, the actual context length will be reduced.")
            # self.context_length = self.context_length - prediction_length # the actual context length
            context_length = context_length - prediction_length

        super(MOIRAI, self).__init__()
        if from_pretrained is not None:
            print(f"Loading model from {from_pretrained}, and ignoring model_args.")
            self.module = MoiraiModule.from_pretrained(from_pretrained)
        else:
            print(f"Initializing model with model_args: {model_args}")
            self.module = MoiraiModule(**model_args)

        # Prepare pre-trained model by downloading model weights from huggingface hub
        self.model = MoiraiForecast(
            module=self.module,
            prediction_length=prediction_length,
            context_length=context_length,
            patch_size=patch_size,
            target_dim=1,
            feat_dynamic_real_dim=None,
            past_feat_dynamic_real_dim=None,
        )

    def _get_batch_samples(self, all_data: torch.Tensor):
        b, l = all_data.size()
        context_len = self.context_length
        required_len = self.context_length + self.prediction_length
        assert l >= required_len, f"Input sequence length {l} is too short, or context length {context_len} is too long."
        random_start = torch.randint(0, l - required_len + 1, (1,)).item()
        sequence = all_data[:, random_start:random_start + required_len]
        context = sequence[:, :context_len]
        target = sequence[:, context_len:]
        return context, target

    def forward(self, all_data: torch.Tensor):
        raise NotImplementedError("MOIRAI is an inference-only model, and does not support training for now.")

    def generate(self, context: torch.Tensor, normalize: bool, reduce: str = None):
        assert reduce in ["mean", "median", None], "Only support mean or median reduce method."

        if normalize:
            mean, std = context.mean(dim=-1, keepdim=True), context.std(dim=-1, keepdim=True)
            std[std == 0] = 1
            context = (context - mean) / std
        
        past_target = context.unsqueeze(-1)
        past_observed_target = torch.ones_like(past_target).bool()
        pasd_is_pad = torch.zeros_like(past_target).squeeze(-1)
        feat_dynamic_real = None
        observed_feat_dynamic_real = None
        
        output = self.model(
            past_target=past_target,
            past_observed_target=past_observed_target,
            past_is_pad=pasd_is_pad,
            feat_dynamic_real=feat_dynamic_real,
            observed_feat_dynamic_real=observed_feat_dynamic_real,
        )

        if normalize:
            mean = mean.unsqueeze(1)
            std = std.unsqueeze(1)
            output = output * std + mean
        
        if reduce == "mean":
            output = output.mean(1)
        elif reduce == "median":
            output = output.median(1).values
        
        return output


if __name__ == "__main__":
    pass
    model = MOIRAI(prediction_length=16, context_length=64, patch_size="auto", from_pretrained="/workspace/S22/TSFM_LLaMA3/huggingface_ckpts/MOIRAI-1.1-Base")
    