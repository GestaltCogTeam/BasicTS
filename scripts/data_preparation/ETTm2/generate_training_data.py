import os
import sys
import argparse

# TODO: remove it when basicts can be installed by pip
sys.path.append(os.path.abspath(__file__ + "/../../../.."))
from scripts.data_preparation.ETTm1.generate_training_data import generate_data


if __name__ == "__main__":
    # sliding window size for generating history sequence and target sequence
    HISTORY_SEQ_LEN = 96
    FUTURE_SEQ_LEN = 336

    TRAIN_RATIO = 0.6
    VALID_RATIO = 0.2
    TARGET_CHANNEL = [0]                   # target channel(s)
    STEPS_PER_DAY = 24 * 4      # every 15 minutes

    DATASET_NAME = "ETTm2"      # sampling frequency: every 1 hour
    TOD = True                  # if add time_of_day feature
    DOW = True                  # if add day_of_week feature
    DOM = True                  # if add day_of_month feature
    DOY = True                  # if add day_of_year feature

    # It is recommended to set norm_each_channel to False when evaluating rescaled data, especially when the magnitudes of different variables differ significantly.
    # Thus, because larger values of the loss function will be obtained when the magnitudes of different variables differ significantly,
    # the model will be trained to focus on the variable with the largest magnitude. Then, the loss is more likely to be reduced.
    # If downstream tasks have other requirements, you can set norm_each_channel to according to your needs.
    NORM_EACH_CHANNEL = False   # if normalize each channel of data separately.

    OUTPUT_DIR = "datasets/" + DATASET_NAME
    DATA_FILE_PATH = "datasets/raw_data/{0}/{0}.csv".format(DATASET_NAME)

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str,
                        default=OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--data_file_path", type=str,
                        default=DATA_FILE_PATH, help="Raw traffic readings.")
    parser.add_argument("--history_seq_len", type=int,
                        default=HISTORY_SEQ_LEN, help="Sequence Length.")
    parser.add_argument("--future_seq_len", type=int,
                        default=FUTURE_SEQ_LEN, help="Sequence Length.")
    parser.add_argument("--steps_per_day", type=int,
                        default=STEPS_PER_DAY, help="Sequence Length.")
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
    args = parser.parse_args()

    # print args
    print("-"*(20+45+5))
    for key, value in sorted(vars(args).items()):
        print("|{0:>20} = {1:<45}|".format(key, str(value)))
    print("-"*(20+45+5))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    generate_data(args)
