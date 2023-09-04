#!/bin/bash
python experiments/train.py -c baselines/MTGNN/MTGNN_METR-LA.py --gpus '3'
python experiments/train.py -c baselines/MTGNN/MTGNN_PEMS-BAY.py --gpus '3'
python experiments/train.py -c baselines/MTGNN/MTGNN_PEMS03.py --gpus '3'
python experiments/train.py -c baselines/MTGNN/MTGNN_PEMS04.py --gpus '3'
python experiments/train.py -c baselines/MTGNN/MTGNN_PEMS07.py --gpus '3'
python experiments/train.py -c baselines/MTGNN/MTGNN_PEMS08.py --gpus '3'
