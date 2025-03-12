# -*-Encoding: utf-8 -*-
################################################################################
#
# Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Description: Some useful metrics
Authors: Lu,Xinjiang (luxinjiang@baidu.com)
Date:    2022/03/10
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import openpyxl as op
import os
import warnings
import random
from sklearn.metrics import r2_score

def rand_spnd(seed=0, fault_num=4, num_sample=100):  # 设置种子为0，确保每次运行程序时都获得相同的随机数序列
    '''

    :param seed: 种子
    :param fault_num: 一轮n个故障spnd
    :return: n组故障样本
    '''
    random.seed(seed)
    ls_rand = []
    numbers = list(range(1, 45))

    for i in range(num_sample):
        selected_numbers = random.sample(numbers, fault_num)
        ls_rand.append(selected_numbers)
        # print(f"第{i + 1}次选择：{selected_numbers}")
    # print('最后的ls:', ls_rand)
    return ls_rand

def mae(pred, gt):
    _mae = 0.
    if len(pred) > 0 and len(gt) > 0:
        _mae = np.mean(np.abs(pred - gt))
    return _mae

def mse(pred, gt):
    _mse = 0.
    if len(pred) > 0 and len(gt) > 0:
        _mse = np.mean((pred - gt)**2)
    return _mse

def rmse(pred, gt):
    return np.sqrt(mse(pred, gt))

def mape(pred, gt):
    _mape = 0.
    # if len(pred) > 0 and len(gt) > 0:
    #     _mape = np.mean(np.abs((pred - gt) / gt)) * 100
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

def mspe(pred, gt):
    return np.mean(np.square((pred - gt) / gt)) if len(pred) > 0 and len(
        gt) > 0 else 0

def add_data(df_path,predict_nd,No_col,sh_name):
    df = pd.read_excel(df_path)
    wb = op.load_workbook(df_path)
    sh = wb[sh_name]
    for i in range(len(df)):
        sh.cell(i+2,No_col,predict_nd[i,0])
    wb.save(df_path)
    return wb

def compute_error(pred, gt):
    ls_ape = []
    for m in range(pred.shape[0]):
        _ape = ((pred[m] - gt[m]) / pred[m]) * 100  # 特殊
        ls_ape.append(_ape)

    return ls_ape

def compute_error_abs(pred, gt):
    '''
    pred (N,T)
    '''
    nd_all_ape = np.zeros(shape=(pred.shape[0],pred.shape[1]))
    for m in range(pred.shape[0]):
        _ape = abs((pred[m] - gt[m]) / (pred[m]+gt[m])) * 200  # 特殊SMAPE
        nd_all_ape[m,:] = _ape.squeeze()
    return nd_all_ape

def evaluate_forecasts_save(predictions, gts, plot_name, config):
    '''
    predictions:(N,F)
    '''
    all_rmse, all_mae, all_mape, all_smape, all_RAE = [], [], [], [], []
    q = 0
    for i in range(config.out_capacity):
        prediction = predictions[:, i]  # (N,1)
        gt = gts[:, i]
        _rmse = rmse(prediction, gt); all_rmse.append(_rmse)
        _mae = mae(prediction, gt); all_mae.append(_mae)
        _mape = mape(prediction, gt); all_mape.append(_mape)
        _smape = smape(prediction, gt); all_smape.append(_smape)
        _rae = RAE(prediction, gt); all_RAE.append(_rae)
        if config.save_flag:
            predict_plot(prediction, gt, plot_name[i], file=config.exp_id, save_flag=config.save_flag)

    if config.save_data_recon_er:
        all_rmse_np = np.array(all_rmse).reshape((-1, 1))
        all_mae_np = np.array(all_mae).reshape((-1, 1))
        all_mape_np = np.array(all_mape).reshape((-1, 1))
        all_smape_np = np.array(all_smape).reshape((-1, 1))
        all_RAE_np = np.array(all_RAE).reshape((-1, 1))
        all_metric = np.concatenate([all_rmse_np, all_mae_np, all_mape_np, all_smape_np, all_RAE_np], axis=1)
        all_metric = pd.DataFrame(all_metric, columns=['RMSE', 'MAE', 'MAPE', 'SMAPE', 'RAE'])
        all_metric.to_csv(f'./save_result/metric/{config.model}_{config.input_len}.csv')
        # 临时
        # if i in save_spnd:
        #     ls_ape = compute_error(prediction, gt)
        #     np_ape = np.array(ls_ape)
        #     np_ape_1_40[:, q] = np_ape.reshape(np_ape_1_40.shape[0], )
        #     q += 1
        #
        #     np_er = ((prediction - gt) / prediction) * 100
        #     out_date = np.concatenate((prediction, gt, np_er), axis=1)
        #     df_out = pd.DataFrame(out_date, columns=['estimated', 'truth', 'ape'])
            # df_out.to_excel(r'.\TM_data\single_save\#{}data.xlsx'.format(str(i+1)))
    # df_ape_1_40 = pd.DataFrame(np_ape_1_40,columns=[str(i+1) for i in save_spnd])
    # df_ape_1_40.to_csv(r'.\TM_data\single_bias_save\#ape_[1-18]_data.csv', index=False)

    # 临时,
    # np_all_mape = np.array(all_mape).reshape(1,-1)
    # df_all_mape = pd.DataFrame(np_all_mape, columns = [str(i) for i in range(1, 45)])
    # df_all_mape.to_csv(r'.\FI_data\#er_1-44_Graph-A.csv', index=False)
    return all_rmse, all_mae, all_mape, all_smape, all_RAE

def evaluate_forecasts_Ex(predictions, gts, plot_name):
    '''
    (N, T)
    '''
    all_rmse, all_mae, all_mape, all_smape = [], [], [], []
    for i in range(len(plot_name)):
        prediction = predictions[i].reshape(-1,1) #(T,1)
        gt = gts[i].reshape(-1,1)
        _rmse = rmse(prediction, gt); all_rmse.append(_rmse)
        _mae = mae(prediction, gt); all_mae.append(_mae)
        _mape = mape(prediction, gt); all_mape.append(_mape)
        _smape = smape(prediction, gt); all_smape.append(_smape)
    return all_rmse, all_mae, all_mape, all_smape


def predict_plot(pred,tru,predict_target, file, save_flag=False,axvline=0,label1='Reconstructed',label2='Fault'):

    x = np.linspace(0, tru.shape[0], tru.shape[0])
    fig = plt.figure(figsize=(10, 5))
    plt.plot(x, pred, label=label1, ls='-.', color='lightskyblue')  # predict
    plt.plot(x, tru, label=label2, ls='--', color='tomato')  # truth
    plt.title(str(predict_target) + ' reconstruction by KGSTN')
    plt.xlabel('Samples')
    plt.ylabel('Current ($\mu$A)')
    plt.legend()
    if save_flag:
        path = f'.\\save_fig\\{file}'
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(f'{path}\\'+str(predict_target))
        plt.clf()
    else:
        plt.axvline(axvline)
        plt.show()

