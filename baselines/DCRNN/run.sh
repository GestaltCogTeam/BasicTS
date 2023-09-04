#!/bin/bash
python experiments/train.py -c baselines/DCRNN/DCRNN_METR-LA.py --gpus '1'
python experiments/train.py -c baselines/DCRNN/DCRNN_PEMS-BAY.py --gpus '1'
python experiments/train.py -c baselines/DCRNN/DCRNN_PEMS03.py --gpus '1'
python experiments/train.py -c baselines/DCRNN/DCRNN_PEMS04.py --gpus '1'
python experiments/train.py -c baselines/DCRNN/DCRNN_PEMS07.py --gpus '1'
python experiments/train.py -c baselines/DCRNN/DCRNN_PEMS08.py --gpus '1'
