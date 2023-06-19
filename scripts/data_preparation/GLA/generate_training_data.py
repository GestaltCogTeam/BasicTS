import os
import sys
import argparse


# TODO: remove it when basicts can be installed by pip
sys.path.append(os.path.abspath(__file__ + "/../../../.."))
from scripts.data_preparation.CA.generate_training_data import generate_data

# Dataset Description: 
#   LargeST: A Benchmark Dataset for Large-Scale Traffic Forecasting.


if __name__ == "__main__":
    # sliding window size for generating history sequence and target sequence
    HISTORY_SEQ_LEN = 12
    FUTURE_SEQ_LEN = 12

    TRAIN_RATIO = 0.6
    VALID_RATIO = 0.2
    TARGET_CHANNEL = [0]                   # target channel(s)

    DATASET_NAME = "GLA"
    TOD = True                  # if add time_of_day feature
    DOW = True                  # if add day_of_week feature
    DOM = True                  # if add day_of_month feature
    DOY = True                  # if add day_of_year feature

    NORM_EACH_CHANNEL = False

    OUTPUT_DIR = "datasets/" + DATASET_NAME
    DATA_FILE_PATH = "datasets/raw_data/{0}/{0}.h5".format(DATASET_NAME)
    GRAPH_FILE_PATH = "datasets/raw_data/{0}/adj_{0}.npy".format(DATASET_NAME)
    GRAPH_METE_PATH = "datasets/raw_data/{0}/meta_{0}.csv".format(DATASET_NAME)

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str,
                        default=OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--data_file_path", type=str,
                        default=DATA_FILE_PATH, help="Raw traffic readings.")
    parser.add_argument("--graph_file_path", type=str,
                        default=GRAPH_FILE_PATH, help="Raw traffic readings.")
    parser.add_argument("--history_seq_len", type=int,
                        default=HISTORY_SEQ_LEN, help="Sequence Length.")
    parser.add_argument("--future_seq_len", type=int,
                        default=FUTURE_SEQ_LEN, help="Sequence Length.")
    parser.add_argument("--tod", type=bool, default=TOD,
                        help="Add feature time_of_day.")
    parser.add_argument("--dow", type=bool, default=DOW,
                        help="Add feature day_of_week.")
    parser.add_argument("--dom", type=bool, default=DOM,
                        help="Add feature day_of_week.")
    parser.add_argument("--doy", type=bool, default=DOY,
                        help="Add feature day_of_week.")
    parser.add_argument("--target_channel", type=list,
                        default=TARGET_CHANNEL, help="Selected channels.")
    parser.add_argument("--train_ratio", type=float,
                        default=TRAIN_RATIO, help="Train ratio")
    parser.add_argument("--valid_ratio", type=float,
                        default=VALID_RATIO, help="Validate ratio.")
    parser.add_argument("--norm_each_channel", type=float,
                        default=NORM_EACH_CHANNEL, help="Validate ratio.")
    args_metr = parser.parse_args()

    # print args
    print("-"*(20+45+5))
    for key, value in sorted(vars(args_metr).items()):
        print("|{0:>20} = {1:<45}|".format(key, str(value)))
    print("-"*(20+45+5))

    if os.path.exists(args_metr.output_dir):
        reply = str(input(
            f"{args_metr.output_dir} exists. Do you want to overwrite it? (y/n)")).lower().strip()
        if reply[0] != "y":
            sys.exit(0)
    else:
        os.makedirs(args_metr.output_dir)
    generate_data(args_metr)
