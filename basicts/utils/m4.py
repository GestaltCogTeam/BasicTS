# This source code is provided for the purposes of scientific reproducibility
# under the following limited license from Element AI Inc. The code is an
# implementation of the N-BEATS model (Oreshkin et al., N-BEATS: Neural basis
# expansion analysis for interpretable time series forecasting,
# https://arxiv.org/abs/1905.10437). The copyright to the source code is
# licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/.  Any commercial use (whether
# for the benefit of third parties or internally in production) requires an
# explicit license. The subject-matter of the N-BEATS model and associated
# materials are the property of Element AI Inc. and may be subject to patent
# protection. No license to patents is granted hereunder (whether express or
# implied). Copyright © 2020 Element AI Inc. All rights reserved.

# Modified from https://github.com/ServiceNow/N-BEATS

"""
M4 Summary
"""
import os
from glob import glob
from dataclasses import dataclass
from collections import OrderedDict

import numpy as np
import pandas as pd

Forecast = np.ndarray
Target = np.ndarray


@dataclass()
class M4Dataset:
    ids: np.ndarray
    groups: np.ndarray
    frequencies: np.ndarray
    horizons: np.ndarray
    values: np.ndarray

    @staticmethod
    def load(info_file_path: str = None, data: np.array = None) -> "M4Dataset":
        """
        Load cached dataset.

        :param training: Load training part if training is True, test part otherwise.
        """
        m4_info = pd.read_csv(info_file_path)
        ids = m4_info.M4id.values
        groups = m4_info.SP.values
        frequencies = m4_info.Frequency.values
        horizons = m4_info.Horizon.values
        values = data
        return M4Dataset(ids=ids, groups=groups, frequencies=frequencies, horizons=horizons, values=values)

def mase(forecast: Forecast, insample: np.ndarray, outsample: Target, frequency: int) -> np.ndarray:
    """
    MASE loss as defined in "Scaled Errors" https://robjhyndman.com/papers/mase.pdf

    :param forecast: Forecast values. Shape: batch, time_o
    :param insample: Insample values. Shape: batch, time_i
    :param outsample: Target values. Shape: batch, time_o
    :param frequency: Frequency value
    :return: Same shape array with error calculated for each time step
    """
    return np.mean(np.abs(forecast - outsample)) / np.mean(np.abs(insample[:-frequency] - insample[frequency:]))


def smape_2(forecast: Forecast, target: Target) -> np.ndarray:
    """
    sMAPE loss as defined in https://robjhyndman.com/hyndsight/smape/ (Makridakis 1993)

    :param forecast: Forecast values. Shape: batch, time
    :param target: Target values. Shape: batch, time
    :return: Same shape array with sMAPE calculated for each time step of each timeseries.
    """
    denom = np.abs(target) + np.abs(forecast)
    # divide by 1.0 instead of 0.0, in case when denom is zero the enumerator will be 0.0 anyway.
    denom[denom == 0.0] = 1.0
    return 200 * np.abs(forecast - target) / denom


def group_values(values: np.ndarray, groups: np.ndarray, group_name: str) -> np.ndarray:
    """
    Filter values array by group indices and clean it from NaNs.

    :param values: Values to filter.
    :param groups: Timeseries groups.
    :param group_name: Group name to filter by.
    :return: Filtered and cleaned timeseries.
    """
    return np.array([v[~np.isnan(v)] for v in values[groups == group_name]], dtype=object)


class M4Summary:
    def __init__(self, info_file_path, train_values, test_values, naive_forecast_file_path):
        self.training_set = M4Dataset.load(info_file_path, train_values)
        self.test_set = M4Dataset.load(info_file_path, test_values)
        self.naive_forecast_file_path = naive_forecast_file_path

    def evaluate(self, forecast: np.ndarray):
        """
        Evaluate forecasts using M4 test dataset.

        :param forecast: Forecasts. Shape: timeseries, time.
        :return: sMAPE and OWA grouped by seasonal patterns.
        """
        forecast = np.array([v[~np.isnan(v)] for v in forecast], dtype=object)

        grouped_smapes = {group_name:
                              np.mean(smape_2(forecast=group_values(values=forecast,
                                                                    groups=self.test_set.groups,
                                                                    group_name=group_name),
                                              target=group_values(values=self.test_set.values,
                                                                  groups=self.test_set.groups,
                                                                  group_name=group_name)))
                          for group_name in np.unique(self.test_set.groups)}
        grouped_smapes = self.summarize_groups(grouped_smapes)

        grouped_owa = OrderedDict()

        naive2_forecasts = pd.read_csv(self.naive_forecast_file_path).values[:, 1:].astype(np.float32)
        naive2_forecasts = np.array([v[~np.isnan(v)] for v in naive2_forecasts], dtype=object)

        model_mases = {}
        naive2_smapes = {}
        naive2_mases = {}
        for group_name in np.unique(self.test_set.groups):
            model_forecast = group_values(forecast, self.test_set.groups, group_name)
            naive2_forecast = group_values(naive2_forecasts, self.test_set.groups, group_name)

            target = group_values(self.test_set.values, self.test_set.groups, group_name)
            # all timeseries within group have same frequency
            frequency = self.training_set.frequencies[self.test_set.groups == group_name][0]
            insample = group_values(self.training_set.values, self.test_set.groups, group_name)

            model_mases[group_name] = np.mean([mase(forecast=model_forecast[i],
                                                    insample=insample[i],
                                                    outsample=target[i],
                                                    frequency=frequency) for i in range(len(model_forecast))])
            naive2_mases[group_name] = np.mean([mase(forecast=naive2_forecast[i],
                                                     insample=insample[i],
                                                     outsample=target[i],
                                                     frequency=frequency) for i in range(len(model_forecast))])

            naive2_smapes[group_name] = np.mean(smape_2(naive2_forecast, target))
        grouped_model_mases = self.summarize_groups(model_mases)
        grouped_naive2_smapes = self.summarize_groups(naive2_smapes)
        grouped_naive2_mases = self.summarize_groups(naive2_mases)
        for k in grouped_model_mases.keys():
            grouped_owa[k] = (grouped_model_mases[k] / grouped_naive2_mases[k] +
                              grouped_smapes[k] / grouped_naive2_smapes[k]) / 2
        def round_all(d):
            return dict(map(lambda kv: (kv[0], np.round(kv[1], 3)), d.items()))
        return round_all(grouped_smapes), round_all(grouped_owa)

    def summarize_groups(self, scores):
        """
        Re-group scores respecting M4 rules.
        :param scores: Scores per group.
        :return: Grouped scores.
        """
        scores_summary = OrderedDict()

        def group_count(group_name):
            return len(np.where(self.test_set.groups == group_name)[0])

        weighted_score = {}
        for g in ["Yearly", "Quarterly", "Monthly"]:
            weighted_score[g] = scores[g] * group_count(g)
            scores_summary[g] = scores[g]

        others_score = 0
        others_count = 0
        for g in ["Weekly", "Daily", "Hourly"]:
            others_score += scores[g] * group_count(g)
            others_count += group_count(g)
        weighted_score["Others"] = others_score
        scores_summary["Others"] = others_score / others_count

        average = np.sum(list(weighted_score.values())) / len(self.test_set.groups)
        scores_summary["Average"] = average

        return scores_summary


def m4_summary(save_dir, project_dir):
    """Summary evaluation for M4 dataset.

    Args:
        save_dir (str): Directory where prediction results are saved. All "{save_dir}/M4_{seasonal pattern}.npy" should exist.
                        Seasonal patterns = ["Yearly", "Quarterly", "Monthly", "Weekly", "Daily", "Hourly"]
        project_dir (str): Project directory. The M4 raw data should be in "{project_dir}/datasets/raw_data/M4".
    """
    seasonal_patterns = ["Yearly", "Quarterly", "Monthly", "Weekly", "Daily", "Hourly"] # the order cannot be changed
    data_dir = project_dir + "/datasets/raw_data/M4"
    info_file_path = data_dir + "/M4-info.csv"

    m4_info = pd.read_csv(info_file_path)
    m4_ids = m4_info.M4id.values
    def build_cache(files: str) -> None:
        timeseries_dict = OrderedDict(list(zip(m4_ids, [[]] * len(m4_ids))))
        for train_csv in glob(os.path.join(data_dir, files)):
            dataset = pd.read_csv(train_csv)
            dataset.set_index(dataset.columns[0], inplace=True)
            for m4id, row in dataset.iterrows():
                values = row.values
                timeseries_dict[m4id] = values[~np.isnan(values)]
        return np.array(list(timeseries_dict.values()), dtype=object)

    print("Building cache for M4 dataset...")
    # read prediction and ground truth
    prediction = []
    for seasonal_pattern in seasonal_patterns:
        prediction.extend(np.load(save_dir + "/M4_{0}.npy".format(seasonal_pattern)))
    prediction = np.array(prediction, dtype=object)
    train_values = build_cache("*-train.csv")
    test_values = build_cache("*-test.csv")
    print("Summarizing M4 dataset...")
    summary = M4Summary(info_file_path, train_values, test_values, data_dir + "/submission-Naive2.csv")
    results = pd.DataFrame(summary.evaluate(prediction), index=["SMAPE", "OWA"])
    return results
