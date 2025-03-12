import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os

# 构建目标目录路径



class FilterHuberLoss(nn.Module):
    def __init__(self, delta=5):
        super(FilterHuberLoss, self).__init__()
        self.delta = delta  # 超参数

    def forward(self, pred, gold):
        return torch.mean(F.smooth_l1_loss(pred, gold, reduction='none', beta=self.delta))


def Anti_testnormalization(pred_y, all_scaler):
    """inverse_transform for each sensor"""
    for s in range(len(all_scaler)):
        np_data = pred_y[:, s]
        scaler = all_scaler[s]
        new_pred = scaler.inverse_transform(np_data)
        pred_y[:, s] = new_pred
    return pred_y

def mae(pred, gt):
    _mae = 0.
    if len(pred) > 0 and len(gt) > 0:
        _mae = np.mean(np.abs(pred - gt))
    return _mae


def mse(pred, gt):
    _mse = 0.
    if len(pred) > 0 and len(gt) > 0:
        _mse = np.mean((pred - gt) ** 2)
    return _mse


def rmse(pred, gt):
    return np.sqrt(mse(pred, gt))


def mape(pred, gt):
    _mape = 0.
    if len(pred) > 0 and len(gt) > 0:
        _mape = np.mean(np.abs((pred - gt) / gt)) * 100
    return _mape


def smape(pred, gt):
    _smape = 0.
    if len(pred) > 0 and len(gt) > 0:
        _smape = 2.0 * np.mean(np.abs(pred - gt) / (np.abs(pred) + np.abs(gt))) * 100
    return _smape


def RAE(pred, gt):
    gt_mean = np.mean(gt)
    squared_error_num = np.sum(np.abs(gt - pred))
    squared_error_den = np.sum(np.abs(gt - gt_mean))
    rae_loss = squared_error_num / squared_error_den
    return rae_loss


def evaluate_forecasts(predictions, gts):
    '''
    predictions:(N,F)
    '''
    all_rmse, all_mae, all_mape, all_smape, all_RAE = [], [], [], [], []
    for i in range(gts.shape[1]):
        prediction = predictions[:, i]  # (N,1)
        gt = gts[:, i]
        _rmse = rmse(prediction, gt);
        all_rmse.append(_rmse)
        _mae = mae(prediction, gt);
        all_mae.append(_mae)
        _mape = mape(prediction, gt);
        all_mape.append(_mape)
        _smape = smape(prediction, gt);
        all_smape.append(_smape)
        _rae = RAE(prediction, gt);
        all_RAE.append(_rae)
    return all_rmse, all_mae, all_mape, all_smape, all_RAE

def upsample(data, rate):
    # data: np.array[len, node_num]
    seq_len, node = data.shape
    new_data = -1 * np.ones([seq_len * rate, node])
    for n in range(node):
        if n == 0:  # 第一列日期，跳过
            continue
        y = data[:, n].astype(np.float32)
        x = np.linspace(0, seq_len, num=seq_len)
        new_x = np.linspace(0, seq_len, num=seq_len * rate)
        new_data[:, n] = np.interp(new_x, x, y)
    return new_data

def plot_save_result(config, gt, pred):
    for i in range(gt.shape[1]):
        # save figure
        plt.figure()
        _gt = gt[:, i]
        _pred = pred[:, i]
        plt.plot(np.arange(len(_gt)), _gt, label='true')
        plt.plot(np.arange(len(_pred)), _pred, label='pred')
        plt.legend()
        save_path = f'./fig/{config.model}/{config.dataset}'
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        plt.savefig(f'{save_path}/P_{i + 1}.png')
        plt.clf()
        plt.close()

        # save csv
        if config.save_pred_data:
            df_data = pd.DataFrame(np.vstack([_gt, _pred]).T,
                                   columns=[f'{i + 1}_true', f'{i + 1}_pred'])
            if i == 0:
                save_df = df_data
            else:
                save_df = pd.concat([save_df, df_data], axis=1)

    if config.save_pred_data:
        target_dir = f'../save_result/data/{config.dataset}'

        # 如果目录不存在，则创建它
        os.makedirs(target_dir, exist_ok=True)
        save_df.to_csv(f'{target_dir}/{config.model}_{config.input_len}.csv', index=False)
