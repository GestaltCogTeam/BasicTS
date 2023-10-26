#!/bin/bash
python experiments/train.py -c baselines/GWNet/METR-LA.py --gpus '1'
python experiments/train.py -c baselines/GWNet/PEMS-BAY.py --gpus '1'
python experiments/train.py -c baselines/GWNet/PEMS03.py --gpus '1'
python experiments/train.py -c baselines/GWNet/PEMS04.py --gpus '1'
python experiments/train.py -c baselines/GWNet/PEMS07.py --gpus '1'
python experiments/train.py -c baselines/GWNet/PEMS08.py --gpus '1'
