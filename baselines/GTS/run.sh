#!/bin/bash
python experiments/train.py -c baselines/GTS/METR-LA.py --gpus '1'
python experiments/train.py -c baselines/GTS/PEMS-BAY.py --gpus '1'
python experiments/train.py -c baselines/GTS/PEMS03.py --gpus '1'
python experiments/train.py -c baselines/GTS/PEMS04.py --gpus '1'
python experiments/train.py -c baselines/GTS/PEMS07.py --gpus '1'
python experiments/train.py -c baselines/GTS/PEMS08.py --gpus '1'
