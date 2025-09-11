#!/bin/bash
python experiments/train.py -c baselines/STEP/STEP_METR-LA.py --gpus '0'
python experiments/train.py -c baselines/STEP/STEP_PEMS-BAY.py --gpus '0'
python experiments/train.py -c baselines/STEP/STEP_PEMS03.py --gpus '0'
python experiments/train.py -c baselines/STEP/STEP_PEMS04.py --gpus '0'
python experiments/train.py -c baselines/STEP/STEP_PEMS07.py --gpus '0'
python experiments/train.py -c baselines/STEP/STEP_PEMS08.py --gpus '0'
