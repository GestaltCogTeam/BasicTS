#!/bin/bash
python experiments/train.py -c baselines/STNorm/STNorm_METR-LA.py --gpus '0'
python experiments/train.py -c baselines/STNorm/STNorm_PEMS-BAY.py --gpus '0'
python experiments/train.py -c baselines/STNorm/STNorm_PEMS03.py --gpus '0'
python experiments/train.py -c baselines/STNorm/STNorm_PEMS04.py --gpus '0'
python experiments/train.py -c baselines/STNorm/STNorm_PEMS07.py --gpus '0'
python experiments/train.py -c baselines/STNorm/STNorm_PEMS08.py --gpus '0'
