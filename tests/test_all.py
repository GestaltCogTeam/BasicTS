import os
import time
import sys

from utils import test

sys.path.append(os.path.abspath(__file__ + "/../.."))

if __name__ == "__main__":
    MODELS = os.listdir("examples")
    MODELS.remove("run.py")
    DATASETS = os.listdir("datasets")
    DATASETS.remove("raw_data")
    DATASETS.remove("README.md")

    with open("test_specific_" + time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()) + ".log", 'w') as f:
        for model in MODELS:
            for dataset in DATASETS:
                test(model, dataset, f)
