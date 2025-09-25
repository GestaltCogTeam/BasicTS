## How to Train MOIRAI
This directory currently supports inference only. Training MOIRAI involves a combination of multiple libraries, including GluonTS and PyTorch Lightning, making it difficult to directly port into BasicTS.

If you are interested in training MOIRAI using the BLAST dataset, please refer to [MOIRAI-BLAST](https://github.com/zezhishao/MOIRAI-BLAST).

## How to Evaluate MOIRAI

To evaluate MOIRAI, you can use the provided configuration files located in the `evaluate_config` directory.
Specifically, you need to download the BLAST-trained models from the [BLAST-CKPTS repository](https://huggingface.co/ZezhiShao/BLAST_CKPTS/tree/main) and modify the `MODEL_PARAM['from_pretrained']` parameter in the evaluating configuration files to point to the downloaded model checkpoints.
Or you can generate the ckpt by yourself using the training code provided in the MOIRAI-BLAST repository.
After setting up the configuration files, you can run the `evaluate_all.py` script to evaluate MOIRAI on all datasets and prediction lengths.
