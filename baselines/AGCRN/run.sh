#!/bin/bash
python experiments/train.py -c baselines/AGCRN/METR-LA.py --gpus '0'
python experiments/train.py -c baselines/AGCRN/PEMS-BAY.py --gpus '0'
python experiments/train.py -c baselines/AGCRN/PEMS03.py --gpus '0'
python experiments/train.py -c baselines/AGCRN/PEMS04.py --gpus '0'
python experiments/train.py -c baselines/AGCRN/PEMS07.py --gpus '0'
python experiments/train.py -c baselines/AGCRN/PEMS08.py --gpus '0'
