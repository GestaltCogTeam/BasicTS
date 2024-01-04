#!/bin/bash
python experiments/train.py -c baselines/GMSDR/METR-LA.py --gpus '4'
python experiments/train.py -c baselines/GMSDR/PEMS-BAY.py --gpus '4'
python experiments/train.py -c baselines/GMSDR/PEMS03.py --gpus '4'
python experiments/train.py -c baselines/GMSDR/PEMS04.py --gpus '4'
python experiments/train.py -c baselines/GMSDR/PEMS07.py --gpus '4'
python experiments/train.py -c baselines/GMSDR/PEMS08.py --gpus '4'