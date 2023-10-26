#!/bin/bash
python experiments/train.py -c baselines/STID/METR-LA.py --gpus '0'
python experiments/train.py -c baselines/STID/PEMS-BAY.py --gpus '0'
python experiments/train.py -c baselines/STID/PEMS03.py --gpus '0'
python experiments/train.py -c baselines/STID/PEMS04.py --gpus '0'
python experiments/train.py -c baselines/STID/PEMS07.py --gpus '0'
python experiments/train.py -c baselines/STID/PEMS08.py --gpus '0'

