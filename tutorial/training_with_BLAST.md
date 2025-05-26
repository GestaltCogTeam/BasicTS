# Training Universal Forecasting Models with the BLAST Dataset

## 1  BLAST Dataset Overview

The [BLAST](https://arxiv.org/abs/2505.17871) dataset is a large-scale time-series corpus created specifically for **Universal Forecasting Models**. Thanks to its rich and diverse samples, BLAST enables **faster convergence**, **notable reductions in computational cost**, and superior performance even with limited resources.  
BasicTS provides native support for BLAST and can be used to train models such as **TimeMoE** (decoder-only architecture) and **ChronosBolt** (encoder-decoder architecture). Before you start, follow the BasicTS [README](../README.md) to clone the codebase and set up the environment.

## 2  Data Preparation

### 2.1  Download the Dataset

Run the following commands from the BasicTS root directory to download BLAST from Hugging Face:

```bash
cd /path/to/BasicTS
huggingface-cli download ZezhiShao/BLAST \
  --repo-type dataset \
  --local-dir ./datasets/BLAST
```

After the download finishes, the data will be under `datasets/BLAST`.

### 2.2  Pre-processing

Because Hugging Face limits single-file sizes, the raw dataset is split into multiple shards. Use the official script to merge them and convert the data into the format expected by BasicTS:

```bash
cd /path/to/BasicTS
python scripts/data_preparation/BLAST/merge_data.py --clean_cache True
```

After merging, the directory becomes:

```
datasets/BLAST
├── train          # training set
│   ├── data.dat   # memory-mapped sequence data
│   └── shape.npy  # (19800000, 4096)
├── valid          # validation set
│   ├── data.dat
│   └── shape.npy  # (200000, 4096)
```

- **`data.dat`**: floating-point sequences stored with NumPy **memory mapping**.  
- **`shape.npy`**: a 2-element NumPy array giving **(number of samples, sequence length)**.

BLAST is an $N \times L$ array, where $N$ is the sample count and $L$ is the sequence length, stored in `float32`.  

> [!IMPORTANT]
> NaN values from the original data are preserved. Handle these NaNs appropriately when loading the data, according to the requirements of your model.  
> See the data loaders for [TimeMoE](../baselines/TimeMoE/data/dataset.py) and [ChronosBolt](../baselines/ChronosBolt/data/dataset.py) for examples.

## 3  Model Training

BasicTS currently supports training **TimeMoE** and **ChronosBolt** on BLAST.

### 3.1  Training ChronosBolt

1. **Download the Pre-trained Weights**

   If you have network restrictions, you can switch to a mirror endpoint (commented in the example):

   ```bash
   # Example for a mainland-China mirror (uncomment to use)
   # export HF_ENDPOINT=https://hf-mirror.com

   cd /path/to/BasicTS

   huggingface-cli download autogluon/chronos-bolt-base \
     --repo-type model \
     --local-dir ./baselines/ChronosBolt/ckpt/ChronosBolt-base/

   huggingface-cli download autogluon/chronos-bolt-small \
     --repo-type model \
     --local-dir ./baselines/ChronosBolt/ckpt/ChronosBolt-small/
   ```

> [!IMPORTANT]
> These checkpoints are provided **solely for convenient random-weight initialization** and are **not** intended for further training on pre-trained weights.

2. **Start Training**

   The example below trains the **Chronos-small** configuration on eight GPUs. To train the base version, uncomment the corresponding line.

   ```bash
   python experiments/train.py \
     -c baselines/ChronosBolt/config/chronos_small.py \
     -g '0,1,2,3,4,5,6,7'

   # python experiments/train.py \
   #   -c baselines/ChronosBolt/config/chronos_base.py \
   #   -g '0,1,2,3,4,5,6,7'
   ```

### 3.2 Training TimeMoE

1. **Download the Pre-trained Weights**

  ```bash
  # If your network is restricted, switch to a mirror first:
  # export HF_ENDPOINT=https://hf-mirror.com

  huggingface-cli download Maple728/TimeMoE-50M  --repo-type model \
    --local-dir ./baselines/TimeMoE/ckpt/TimeMoE-50M

  huggingface-cli download Maple728/TimeMoE-200M --repo-type model \
    --local-dir ./baselines/TimeMoE/ckpt/TimeMoE-200M
  ```

2. **Start Training**

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

> [!TIP]
> - The default config is tuned for BLAST (batch size, learning rate, and other hyper-parameters). Adjust them if your GPUs or hardware differ.  

If you encounter any issues, feel free to open an issue on [BasicTS Issues](https://github.com/GestaltCogTeam/BasicTS/issues). Happy training!