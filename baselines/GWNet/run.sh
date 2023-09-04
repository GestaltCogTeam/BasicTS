#!/bin/bash
python experiments/train.py -c baselines/GWNet/GWNet_METR-LA.py --gpus '3'
python experiments/train.py -c baselines/GWNet/GWNet_PEMS-BAY.py --gpus '3'
python experiments/train.py -c baselines/GWNet/GWNet_PEMS03.py --gpus '3'
python experiments/train.py -c baselines/GWNet/GWNet_PEMS04.py --gpus '3'
python experiments/train.py -c baselines/GWNet/GWNet_PEMS07.py --gpus '3'
python experiments/train.py -c baselines/GWNet/GWNet_PEMS08.py --gpus '3'
