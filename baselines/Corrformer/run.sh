# #!/bin/bash
python experiments/train.py -c baselines/Corrformer/ETTh1.py --gpus '1'
python experiments/train.py -c baselines/Corrformer/Electricity.py --gpus '1'
python experiments/train.py -c baselines/Corrformer/PEMS04.py --gpus '1'
python experiments/train.py -c baselines/Corrformer/PEMS08.py --gpus '1'
