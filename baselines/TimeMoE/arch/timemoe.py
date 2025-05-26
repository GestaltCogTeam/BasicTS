import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM

from .configuration_time_moe import TimeMoeConfig
from .modeling_time_moe import TimeMoeForPrediction

class TimeMoE(nn.Module):
    def __init__(self, model_id: str, from_pretrained: bool,
                    context_length: int,
                    trust_remote_code: bool):
        
        super().__init__()
        
        self.model_type = 'causal' # TimeMoE is a causal model
        self.context_length = context_length
        
        if from_pretrained:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=trust_remote_code,
            )
        else:
            kwargs = {}
            kwargs['torch_dtype'] = 'float32'
            kwargs['attn_implementation'] = 'flash_attention_2'
            config, model_kwargs = TimeMoeConfig.from_pretrained(
                                    pretrained_model_name_or_path=model_id,
                                    return_unused_kwargs=True,
                                    **kwargs)
            print(f'Using attention implementation: {kwargs.get("attn_implementation", "original")}')
            self.model = TimeMoeForPrediction(config)

    def forward(self, context: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):

        output = self.model(input_ids=context, labels=target, loss_masks=mask) # NOTE: TimeMoE uses Huber loss

        loss, _ = output.loss, output.logits # _ is the logits

        return loss

    def generate(self, context: torch.Tensor, prediction_length: int, normalize: bool = False):
        if normalize:
            mean, std = context.mean(dim=-1, keepdim=True), context.std(dim=-1, keepdim=True)
            std[std == 0] = 1
            context = (context - mean) / std
        
        predictions = self.model.generate(
            input_ids=context,
            max_new_tokens=prediction_length
        )
        predictions = predictions[:, -prediction_length:]

        if normalize:
            predictions = predictions * std + mean

        return predictions
