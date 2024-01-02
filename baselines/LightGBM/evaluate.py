import torch
import lightgbm as lgb
import os
import sys
sys.path.append("/workspace/S22/BasicTS")
from basicts.utils import load_pkl
from basicts.data import TimeSeriesForecastingDataset
from basicts.metrics import masked_mae, masked_rmse, masked_mape, masked_wape
from basicts.data import SCALER_REGISTRY


def evaluate(project_dir, train_data_dir, input_len, output_len, rescale, null_val, batch_size):
        
    # construct dataset
    data_file_path = project_dir + "/{0}/data_in_{1}_out_{2}_rescale_{3}.pkl".format(train_data_dir, input_len, output_len, rescale)
    index_file_path = project_dir + "/{0}/index_in_{1}_out_{2}_rescale_{3}.pkl".format(train_data_dir, input_len, output_len, rescale)

    train_set = TimeSeriesForecastingDataset(data_file_path, index_file_path, mode="train")
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    valid_set = TimeSeriesForecastingDataset(data_file_path, index_file_path, mode="valid")
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False)

    test_set = TimeSeriesForecastingDataset(data_file_path, index_file_path, mode="test")
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
        
    # training & validation
    Xs_train = []
    Ys_train = []
    Xs_valid = []
    Ys_valid = []
    Xs_test = []
    Ys_test = []

    for i, (target, data) in enumerate(train_loader):
        B, L, N, C = data.shape
        data = data.transpose(1, 2).reshape(B*N, L, C)[:, :, 0]
        target = target.transpose(1, 2).reshape(B*N, L, C)[:, :, 0]
        Xs_train.append(data)
        Ys_train.append(target)

    for i, (target, data) in enumerate(valid_loader):
        B, L, N, C = data.shape
        data = data.transpose(1, 2).reshape(B*N, L, C)[:, :, 0]
        target = target.transpose(1, 2).reshape(B*N, L, C)[:, :, 0]
        Xs_valid.append(data)
        Ys_valid.append(target)

    for i, (target, data) in enumerate(test_loader):
        B, L, N, C = data.shape
        data = data.transpose(1, 2).reshape(B*N, L, C)[:, :, 0]
        target = target.transpose(1, 2).reshape(B*N, L, C)[:, :, 0]
        Xs_test.append(data)
        Ys_test.append(target)

    Xs_train = torch.cat(Xs_train, dim=0).numpy()
    Ys_train = torch.cat(Ys_train, dim=0).numpy()
    Xs_valid = torch.cat(Xs_valid, dim=0).numpy()
    Ys_valid = torch.cat(Ys_valid, dim=0).numpy()
    Xs_test = torch.cat(Xs_test, dim=0).numpy()
    Ys_test = torch.cat(Ys_test, dim=0).numpy()

    # direct forecasting
    from sklearn.multioutput import MultiOutputRegressor
    model = MultiOutputRegressor(lgb.LGBMRegressor(), n_jobs = -1)
    model.fit(Xs_train, Ys_train)
    # inference
    preds_test = model.predict(Xs_test)
    print(preds_test.shape)
    # rescale
    scaler = load_pkl(project_dir + "/{0}/scaler_in_{1}_out_{2}_rescale_{3}.pkl".format(train_data_dir, input_len, output_len, rescale))
    preds_test = torch.Tensor(preds_test).view(-1, N, output_len).transpose(1, 2).unsqueeze(-1)
    Ys_test = torch.Tensor(Ys_test).view(-1, N, output_len).transpose(1, 2).unsqueeze(-1)
    prediction = SCALER_REGISTRY.get(scaler["func"])(preds_test, **scaler["args"])
    real_value = SCALER_REGISTRY.get(scaler["func"])(Ys_test, **scaler["args"])
    # print results
    print("MAE: ", masked_mae(prediction, real_value, null_val).item())
    print("RMSE: ", masked_rmse(prediction, real_value, null_val).item())
    print("MAPE: {:.2f}%".format(masked_mape(prediction, real_value, null_val) * 100))
    print("WAPE: {:.2f}%".format(masked_wape(prediction, real_value, null_val) * 100))
