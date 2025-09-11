# -*- coding:utf-8 -*-
import torch
import numpy as np

def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    #print(torch.isnan(mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)
def masked_mse_np(y_pred, y_true, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mse = np.square(np.subtract(y_pred, y_true).astype('float32'))
        mse = np.nan_to_num(mask * mse)
        return np.mean(mse)
def masked_rmse_np(preds, labels, null_val=np.nan):
    return np.sqrt(masked_mse_np(y_pred=preds, y_true=labels, null_val=null_val))
def masked_mae_np(y_pred, y_true, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(y_pred, y_true).astype('float32'))
        mae = np.nan_to_num(mask * mae)
        return np.mean(mae)
def masked_mape_np(y_true, y_pred, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(y_pred, y_true).astype('float32'),
                      y_true))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100

def masked_loss_np(y_pred, y_true, loss_type):
    mask = np.greater(y_true, 1e-1).astype(np.float)
    mask = np.divide(mask, np.mean(mask))
    if loss_type == "mae":
        loss = np.abs(np.subtract(y_pred, y_true))
    elif loss_type == "mape":
        loss = np.divide(np.abs(np.subtract(y_pred, y_true)), y_true)
    elif loss_type == "rmse":
        loss = np.power(np.subtract(y_pred, y_true), 2)
    else:
        raise ValueError("No Such Loss!")
    loss = np.nan_to_num(np.multiply(loss, mask))
    return np.mean(loss)
