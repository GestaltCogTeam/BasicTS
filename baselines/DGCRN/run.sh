#!/bin/bash
python experiments/train.py -c baselines/DGCRN/DGCRN_METR-LA.py --gpus '2'
python experiments/train.py -c baselines/DGCRN/DGCRN_PEMS-BAY.py --gpus '2'
python experiments/train.py -c baselines/DGCRN/DGCRN_PEMS03.py --gpus '2'
python experiments/train.py -c baselines/DGCRN/DGCRN_PEMS04.py --gpus '2'
python experiments/train.py -c baselines/DGCRN/DGCRN_PEMS07.py --gpus '2'
python experiments/train.py -c baselines/DGCRN/DGCRN_PEMS08.py --gpus '2'
