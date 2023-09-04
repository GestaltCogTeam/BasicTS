#!/bin/bash
python experiments/train.py -c baselines/StemGNN/StemGNN_METR-LA.py --gpus '0'
python experiments/train.py -c baselines/StemGNN/StemGNN_PEMS-BAY.py --gpus '0'
python experiments/train.py -c baselines/StemGNN/StemGNN_PEMS03.py --gpus '0'
python experiments/train.py -c baselines/StemGNN/StemGNN_PEMS04.py --gpus '0'
python experiments/train.py -c baselines/StemGNN/StemGNN_PEMS07.py --gpus '0'
python experiments/train.py -c baselines/StemGNN/StemGNN_PEMS08.py --gpus '0'
