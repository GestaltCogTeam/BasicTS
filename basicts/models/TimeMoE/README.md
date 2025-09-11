**TimeMoE** relies on [FlashAttention](https://github.com/Dao-AILab/flash-attention/releases). The version verified in this project is `flash_attn-2.7.4`:

```bash
# Activate your conda environment if applicable
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
pip install flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
```

> [!TIP]
> Download the wheel that matches **your CUDA, PyTorch version, and system architecture** before installation.

If your **GPU memory is limited**, set `CFG.TRAIN.COMPILE_MODEL=True` in the configuration file and add the following lines at the top of the import section in `experiments/train.py` to prevent potential errors and speed up training:

```python
import torch
import torch._dynamo

torch.set_float32_matmul_precision("high")
torch._dynamo.config.accumulated_cache_size_limit = 256
torch._dynamo.config.cache_size_limit = 256  # Increase if necessary
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.optimize_ddp = False
```