# 使用 BLAST 数据集训练通用预测模型

## 1. BLAST 数据集概述

[BLAST](https://arxiv.org/abs/2505.17871) 数据集是一套专为训练 **通用预测模型**（Universal Forecasting Models）而构建的大规模时间序列语料库。借助其丰富且多样的序列样本，模型可以 **更快收敛**、**显著降低计算成本**，并在有限资源下取得更优性能。  
BasicTS 已原生集成对 BLAST 的支持，可用于训练 **TimeMoE**（Decoder-only 架构）与 **ChronosBolt**（Encoder-Decoder 架构）等模型。开始之前，请先按照 BasicTS [README](../README.md) 完成代码拉取与环境配置。

## 2. 数据准备

### 2.1 数据集下载

在本地 BasicTS 根目录执行下列命令，从 Hugging Face 下载 BLAST 数据集：

```bash
cd /path/to/BasicTS
huggingface-cli download ZezhiShao/BLAST \
  --repo-type dataset \
  --local-dir ./datasets/BLAST
```

下载完成后，数据位于 `datasets/BLAST`。

### 2.2 数据预处理

由于 Hugging Face 对单文件大小有限制，原始数据被拆分为多个分片。使用官方脚本合并并转换为 BasicTS 所需格式：

```bash
cd /path/to/BasicTS
python scripts/data_preparation/BLAST/merge_data.py --clean_cache True
```

合并完成后，目录结构更新为：

```
datasets/BLAST
├── train        # 训练集
│   ├── data.dat     # mmap 存储的序列数据
│   └── shape.npy    # (19800000, 4096)
├── valid        # 验证集
│   ├── data.dat
│   └── shape.npy    # (200000, 4096)
```

- `data.dat`：采用 NumPy **内存映射**格式存储的浮点序列。  
- `shape.npy`：对应数据的二维形状数组，格式为 **(样本数, 序列长度)**。

BLAST是一个 $N\times L$的数组，其中 $N$ 是样本数量，$L$ 是序列长度，数据类型为 `float32`。

> [!IMPORTANT]
> 需要注意的是，BLAST保留了原始数据中的NaN值。在读取数据时，你应该根据模型的要求合理地处理这些NaN值。其读取方式可见[TimeMoE](../baselines/TimeMoE/data/dataset.py)和[Chronos](../baselines/ChronosBolt/data/dataset.py)。

## 3. 模型训练

BasicTS 当前支持使用 BLAST 训练TimeMoE和ChronosBolt两种模型。

### 3.1 训练 ChronosBolt

1. **下载预训练权重**

   如遇网络受限，可切换到镜像端点（示例已给出注释）：

   ```bash
   # 国内镜像示例（取消注释后使用）
   # export HF_ENDPOINT=https://hf-mirror.com
   cd /path/to/BasicTS

   huggingface-cli download autogluon/chronos-bolt-base \
     --repo-type model \
     --local-dir ./baselines/ChronosBolt/ckpt/chronos-bolt-base/

   huggingface-cli download autogluon/chronos-bolt-small \
     --repo-type model \
     --local-dir ./baselines/ChronosBolt/ckpt/chronos-bolt-small/
   ```

> [!IMPORTANT]
> 以上 checkpoint 仅用于快速初始化随机权重，并非在已有预训练权重上继续训练。

2. **启动训练**

   以下示例以 **Chronos-small** 配置在 8 张 GPU 上训练；如需训练 base 版本，将对应行取消注释即可。

   ```bash
   python experiments/train.py \
     -c baselines/ChronosBolt/config/chronos_small.py \
     -g '0,1,2,3,4,5,6,7'

   # python experiments/train.py \
   #   -c baselines/ChronosBolt/config/chronos_base.py \
   #   -g '0,1,2,3,4,5,6,7'
   ```

### 3.2 训练 TimeMoE

1. **下载预训练权重**

   ```bash
   # 国内镜像示例（取消注释后使用）
   # export HF_ENDPOINT=https://hf-mirror.com
   cd /path/to/BasicTS

   huggingface-cli download Maple728/TimeMoE-50M  --repo-type model \
    --local-dir ./baselines/TimeMoE/ckpt/TimeMoE-50M

   huggingface-cli download Maple728/TimeMoE-200M --repo-type model \
    --local-dir ./baselines/TimeMoE/ckpt/TimeMoE-200M
   ```

2. **启动训练**

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
> - 默认配置已针对 BLAST 数据集做了批量大小、学习率等超参适配，若显存或机器规格不同，请适当调整。  

如有任何问题，欢迎在 [BasicTS Issues](https://github.com/GestaltCogTeam/BasicTS/issues) 提交反馈。祝训练顺利！