#!/bin/bash
python experiments/train.py -c baselines/StemGNN/METR-LA.py --gpus '0'
python experiments/train.py -c baselines/StemGNN/PEMS-BAY.py --gpus '0'
python experiments/train.py -c baselines/StemGNN/PEMS03.py --gpus '0'
python experiments/train.py -c baselines/StemGNN/PEMS04.py --gpus '0'
python experiments/train.py -c baselines/StemGNN/PEMS07.py --gpus '0'
python experiments/train.py -c baselines/StemGNN/PEMS08.py --gpus '0'
