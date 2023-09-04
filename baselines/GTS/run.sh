#!/bin/bash
python experiments/train.py -c baselines/GTS/GTS_METR-LA.py --gpus '3'
python experiments/train.py -c baselines/GTS/GTS_PEMS-BAY.py --gpus '3'
python experiments/train.py -c baselines/GTS/GTS_PEMS03.py --gpus '3'
python experiments/train.py -c baselines/GTS/GTS_PEMS04.py --gpus '3'
python experiments/train.py -c baselines/GTS/GTS_PEMS07.py --gpus '3'
python experiments/train.py -c baselines/GTS/GTS_PEMS08.py --gpus '3'
