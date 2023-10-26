#!/bin/bash
python experiments/train.py -c baselines/DGCRN/METR-LA.py --gpus '2'
python experiments/train.py -c baselines/DGCRN/PEMS-BAY.py --gpus '2'
python experiments/train.py -c baselines/DGCRN/PEMS03.py --gpus '2'
python experiments/train.py -c baselines/DGCRN/PEMS04.py --gpus '2'
python experiments/train.py -c baselines/DGCRN/PEMS07.py --gpus '2'
python experiments/train.py -c baselines/DGCRN/PEMS08.py --gpus '2'
