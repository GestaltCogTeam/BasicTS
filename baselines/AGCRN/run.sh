#!/bin/bash
python experiments/train.py -c baselines/AGCRN/AGCRN_METR-LA.py --gpus '0'
python experiments/train.py -c baselines/AGCRN/AGCRN_PEMS-BAY.py --gpus '0'
python experiments/train.py -c baselines/AGCRN/AGCRN_PEMS03.py --gpus '0'
python experiments/train.py -c baselines/AGCRN/AGCRN_PEMS04.py --gpus '0'
python experiments/train.py -c baselines/AGCRN/AGCRN_PEMS07.py --gpus '0'
python experiments/train.py -c baselines/AGCRN/AGCRN_PEMS08.py --gpus '0'
