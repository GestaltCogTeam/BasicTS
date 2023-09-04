#!/bin/bash
python experiments/train.py -c baselines/STGCN/STGCN_METR-LA.py --gpus '0'
python experiments/train.py -c baselines/STGCN/STGCN_PEMS-BAY.py --gpus '0'
python experiments/train.py -c baselines/STGCN/STGCN_PEMS03.py --gpus '0'
python experiments/train.py -c baselines/STGCN/STGCN_PEMS04.py --gpus '0'
python experiments/train.py -c baselines/STGCN/STGCN_PEMS07.py --gpus '0'
python experiments/train.py -c baselines/STGCN/STGCN_PEMS08.py --gpus '0'
