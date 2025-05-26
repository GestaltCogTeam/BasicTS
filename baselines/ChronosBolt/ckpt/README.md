**Download ChronosBolt from Hugging Face:**

```bash
# If your network is restricted, switch to a mirror first:
# export HF_ENDPOINT=https://hf-mirror.com

huggingface-cli download autogluon/chronos-bolt-base  \
  --repo-type model \
  --local-dir ./baselines/ChronosBolt/ckpt/chronos-bolt-base/

huggingface-cli download autogluon/chronos-bolt-small \
  --repo-type model \
  --local-dir ./baselines/ChronosBolt/ckpt/chronos-bolt-small/
```

> [!IMPORTANT]
> These checkpoints are provided **solely for convenient random-weight initialization** and are **not** intended for further training on pre-trained weights.

**Train the model:**

```bash
# Train ChronosBolt-small on 8 GPUs
python experiments/train.py \
  -c baselines/ChronosBolt/config/chronos_small.py \
  -g '0,1,2,3,4,5,6,7'

# To train the base variant, uncomment the line below
# python experiments/train.py \
#   -c baselines/ChronosBolt/config/chronos_base.py \
#   -g '0,1,2,3,4,5,6,7'
```