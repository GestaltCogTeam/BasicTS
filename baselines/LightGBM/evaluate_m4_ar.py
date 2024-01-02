import torch
import lightgbm as lgb
import os
import sys
sys.path.append("/workspace/S22/BasicTS")
import numpy as np
from tqdm import tqdm
from basicts.utils import load_pkl
from basicts.data import M4ForecastingDataset
from basicts.metrics import masked_mae, masked_rmse, masked_mape, masked_wape
from basicts.data import SCALER_REGISTRY


def evaluate(project_dir, train_data_dir, input_len, output_len, rescale, null_val, batch_size, patch_len, down_sampling=1, seasonal_pattern=None):
    assert output_len % patch_len == 0
    num_steps = output_len // patch_len
    # construct dataset
    data_file_path = project_dir + "/{0}/data_in_{1}_out_{2}_rescale_{3}.pkl".format(train_data_dir, input_len, output_len, rescale)
    mask_file_path = project_dir + "/{0}/mask_in_{1}_out_{2}_rescale_{3}.pkl".format(train_data_dir, input_len, output_len, rescale)
    index_file_path = project_dir + "/{0}/index_in_{1}_out_{2}_rescale_{3}.pkl".format(train_data_dir, input_len, output_len, rescale)

    train_set = M4ForecastingDataset(data_file_path, index_file_path, mask_file_path, mode="train")
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    test_set = M4ForecastingDataset(data_file_path, index_file_path, mask_file_path, mode="test")
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # training & validation
    Xs_train = []
    Ys_train = []
    Xs_test = []
    Ys_test = []

    for i, (target, data, future_mask, history_mask) in enumerate(train_loader):
        B, L, N, C = data.shape
        data = data.transpose(1, 2).reshape(B*N, L, C)[:, :, 0]
        B, L, N, C = target.shape
        target = target.transpose(1, 2).reshape(B*N, L, C)[:, :, 0]
        Xs_train.append(data)
        Ys_train.append(target)

    Xs_train = torch.cat(Xs_train, dim=0).numpy()[::down_sampling, :]
    Ys_train = torch.cat(Ys_train, dim=0).numpy()[::down_sampling, :][:, :patch_len]
    print("Xs_train: ", Xs_train.shape)

    # direct forecasting
    from sklearn.multioutput import MultiOutputRegressor
    model = MultiOutputRegressor(lgb.LGBMRegressor(), n_jobs = -1)
    model.fit(Xs_train, Ys_train)

    for i, (target, data, future_mask, history_mask) in enumerate(test_loader):
        B, L, N, C = data.shape
        data = data.transpose(1, 2).reshape(B*N, L, C)[:, :, 0]
        B, L, N, C = target.shape
        target = target.transpose(1, 2).reshape(B*N, L, C)[:, :, 0]
        Xs_test.append(data)
        Ys_test.append(target)

    Xs_test = torch.cat(Xs_test, dim=0).numpy()
    Ys_test = torch.cat(Ys_test, dim=0).numpy()
    print("Xs_test: ", Xs_test.shape)

    # inference
    preds_test = []
    input_data = Xs_test

    for i in tqdm(range(num_steps)):
        # Predict the next step
        pred_step = model.predict(input_data)
        preds_test.append(pred_step)
        # Update input_data to include predicted step for next prediction
        input_data = np.concatenate([input_data[:, patch_len:], pred_step[:, :]], axis=1)
    # concat preds_test
    # preds_test = np.vstack(preds_test).T
    preds_test = np.concatenate(preds_test, axis=1)

    # rescale
    preds_test = torch.Tensor(preds_test).view(-1, N, output_len).transpose(1, 2).unsqueeze(-1)
    Ys_test = torch.Tensor(Ys_test).view(-1, N, output_len).transpose(1, 2).unsqueeze(-1)
    prediction = preds_test
    real_value = Ys_test
    np.save("/workspace/S22/BasicTS/baselines/LightGBM/M4_{0}.npy".format(seasonal_pattern), prediction.unsqueeze(-1).unsqueeze(-1).numpy())

    # print results
    print("MAE: ", masked_mae(prediction, real_value, null_val).item())
    print("RMSE: ", masked_rmse(prediction, real_value, null_val).item())
    print("MAPE: {:.2f}%".format(masked_mape(prediction, real_value, null_val) * 100))
    print("WAPE: {:.2f}%".format(masked_wape(prediction, real_value, null_val) * 100))
    # save
    