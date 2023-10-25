#!/bin/bash
python experiments/train.py -c baselines/STAEformer/STAEformer_METR-LA.py --gpus '0'
python experiments/train.py -c baselines/STAEformer/STAEformer_PEMS-BAY.py --gpus '0'
python experiments/train.py -c baselines/STAEformer/STAEformer_PEMS03.py --gpus '0'
python experiments/train.py -c baselines/STAEformer/STAEformer_PEMS04.py --gpus '0'
python experiments/train.py -c baselines/STAEformer/STAEformer_PEMS07.py --gpus '0'
python experiments/train.py -c baselines/STAEformer/STAEformer_PEMS08.py --gpus '0'
