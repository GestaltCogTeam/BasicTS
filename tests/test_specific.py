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

    print("Current support models: {0}".format(MODELS))
    print("Current support datasets: {0}".format(DATASETS))

    reply_models = str(input(f"Please select the names of the models to test, separated by commas: "))
    if reply_models == "":
        reply_models = MODELS
    else:
        reply_models = reply_models.strip().replace(" ", "").split(",")

    reply_datasets = str(input(f"Please select the names of the datasets to test, separated by commas: "))
    if reply_datasets == "":
        reply_datasets = DATASETS
    else:
        reply_datasets = reply_datasets.strip().replace(" ", "").split(",")

    print("Models to test: {0}".format(reply_models))
    print("Datasets to test: {0}".format(reply_datasets))

    with open("test_specific_" + time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()) + ".log", 'w') as f:
        for model in reply_models:
            for dataset in reply_datasets:
                test(model, dataset, f)
