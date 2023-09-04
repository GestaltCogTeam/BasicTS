#!/bin/bash
python experiments/train.py -c baselines/D2STGNN/D2STGNN_METR-LA.py --gpus '2'
python experiments/train.py -c baselines/D2STGNN/D2STGNN_PEMS-BAY.py --gpus '2'
python experiments/train.py -c baselines/D2STGNN/D2STGNN_PEMS03.py --gpus '2'
python experiments/train.py -c baselines/D2STGNN/D2STGNN_PEMS04.py --gpus '2'
python experiments/train.py -c baselines/D2STGNN/D2STGNN_PEMS07.py --gpus '2'
python experiments/train.py -c baselines/D2STGNN/D2STGNN_PEMS08.py --gpus '2'
