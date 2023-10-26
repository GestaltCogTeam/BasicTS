#!/bin/bash
python experiments/train.py -c baselines/MTGNN/METR-LA.py --gpus '3'
python experiments/train.py -c baselines/MTGNN/PEMS-BAY.py --gpus '3'
python experiments/train.py -c baselines/MTGNN/PEMS03.py --gpus '3'
python experiments/train.py -c baselines/MTGNN/PEMS04.py --gpus '3'
python experiments/train.py -c baselines/MTGNN/PEMS07.py --gpus '3'
python experiments/train.py -c baselines/MTGNN/PEMS08.py --gpus '3'
