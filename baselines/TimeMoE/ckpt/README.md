**Download TimeMoE from Hugging Face:**

```bash
# If your network is restricted, switch to a mirror first:
# export HF_ENDPOINT=https://hf-mirror.com

huggingface-cli download Maple728/TimeMoE-50M  --repo-type model \
  --local-dir ./baselines/TimeMoE/ckpt/TimeMoE-50M

huggingface-cli download Maple728/TimeMoE-200M --repo-type model \
  --local-dir ./baselines/TimeMoE/ckpt/TimeMoE-200M
```

> [!IMPORTANT]
> These checkpoints are provided **solely for convenient random-weight initialization** and are **not** intended for further training on pre-trained weights.

**Train the model:**

```bash
# Train TimeMoE-base on 8 GPUs
python experiments/train.py \
  -c baselines/TimeMoE/config/timemoe_base.py \
  -g '0,1,2,3,4,5,6,7'

# To train the large variant, uncomment the line below
# python experiments/train.py \
#   -c baselines/TimeMoE/config/timemoe_large.py \
#   -g '0,1,2,3,4,5,6,7'
```
