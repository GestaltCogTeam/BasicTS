#!/bin/bash
python experiments/train.py -c baselines/STNorm/METR-LA.py --gpus '0'
python experiments/train.py -c baselines/STNorm/PEMS-BAY.py --gpus '0'
python experiments/train.py -c baselines/STNorm/PEMS03.py --gpus '0'
python experiments/train.py -c baselines/STNorm/PEMS04.py --gpus '0'
python experiments/train.py -c baselines/STNorm/PEMS07.py --gpus '0'
python experiments/train.py -c baselines/STNorm/PEMS08.py --gpus '0'
