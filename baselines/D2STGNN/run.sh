#!/bin/bash
python experiments/train.py -c baselines/D2STGNN/METR-LA.py --gpus '2'
python experiments/train.py -c baselines/D2STGNN/PEMS-BAY.py --gpus '2'
python experiments/train.py -c baselines/D2STGNN/PEMS03.py --gpus '2'
python experiments/train.py -c baselines/D2STGNN/PEMS04.py --gpus '2'
python experiments/train.py -c baselines/D2STGNN/PEMS07.py --gpus '2'
python experiments/train.py -c baselines/D2STGNN/PEMS08.py --gpus '2'
