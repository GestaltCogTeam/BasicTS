#!/bin/bash
python experiments/train.py -c baselines/DCRNN/METR-LA.py --gpus '1'
python experiments/train.py -c baselines/DCRNN/PEMS-BAY.py --gpus '1'
python experiments/train.py -c baselines/DCRNN/PEMS03.py --gpus '1'
python experiments/train.py -c baselines/DCRNN/PEMS04.py --gpus '1'
python experiments/train.py -c baselines/DCRNN/PEMS07.py --gpus '1'
python experiments/train.py -c baselines/DCRNN/PEMS08.py --gpus '1'
