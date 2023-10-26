#!/bin/bash
python experiments/train.py -c baselines/STGCN/METR-LA.py --gpus '0'
python experiments/train.py -c baselines/STGCN/PEMS-BAY.py --gpus '0'
python experiments/train.py -c baselines/STGCN/PEMS03.py --gpus '0'
python experiments/train.py -c baselines/STGCN/PEMS04.py --gpus '0'
python experiments/train.py -c baselines/STGCN/PEMS07.py --gpus '0'
python experiments/train.py -c baselines/STGCN/PEMS08.py --gpus '0'
