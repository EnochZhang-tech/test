import torch
import numpy as np
import pandas as pd
import argparse
import random
import os
from sklearn.preprocessing import StandardScaler
from easydict import EasyDict as edict
from utils import plot_save_result, evaluate_forecasts, Anti_testnormalization, upsample
from dataset import dataset, ML_dataset
from matplotlib import pyplot as plt
from PST_utils import one_zero_normalization
from models import *
from collections import deque

def set_seed(seed):
    """
    Set seed for reproduction.
    传入的数值用于指定随机数生成时所用算法开始时所选定的整数值，如果使用相同的seed()值，则每次生成的随机数都相同；
    如果不设置这个值，则系统会根据时间来自己选择这个值，此时每次生成的随机数会因时间的差异而有所不同。
    """
    random.seed(seed)  # random模块的随机数种子
    np.random.seed(seed)  # np.random的随机数种子
    torch.manual_seed(seed)  # 为CPU设置随机数种子,對精度影響不大
    torch.cuda.manual_seed_all(seed)  # 为GPU设置随机数种子
    torch.backends.cudnn.deterministic = True  # 使用cpu时不需要设置，使用gpu时需要设置
    # 顾名思义，将这个 flag 置为True的话，每次返回的卷积算法将是确定的，即默认算法。
    # 如果配合上设置 Torch 的随机种子为固定值的话，应该可以保证每次运行网络的时候相同输入的输出是固定的


def get_model(config,seed,n,c):
    if config.model not in globals():
        raise NotImplementedError("Not found the model: {}".format(config.model))
    model = globals()[config.model](config,seed,n,c)
    return model


def get_data(config):
    train_data = pd.read_csv(config.train_file)
    train_data = train_data.dropna()
    # if config.dataset == 'test_2':  # 数据集2部分数据上采样
    #     num_point = 500
    #     np_low_data = np.array(train_data.iloc[-num_point:, :])
    #     new_np_data = upsample(np_low_data, rate=5)  # 5倍上采样
    #     new_df_data = pd.DataFrame(new_np_data, columns=train_data.columns)
    #     train_data = pd.concat([train_data.iloc[:-num_point, :], new_df_data], axis=0, ignore_index=True)

    train_downsample = np.arange(0, len(train_data), config.train_downsample)
    # train_data = train_data.iloc[train_downsample, 1:24 + 1]  # downsample of train data
    train_data = train_data.iloc[train_downsample, 1:1+4]  # downsample of train data
    np_train_data = np.array(train_data)

    test_data = pd.read_csv(config.test_file)
    test_data = test_data.dropna()
    test_downsample = np.arange(0, len(test_data), config.test_downsample)
    # test_data = test_data.iloc[test_downsample, 1:24 + 1]  # downsample of test data
    test_data = test_data.iloc[test_downsample, :]  # downsample of test data
    test_date_list = list(pd.to_datetime(test_data['datetime']))
    test_data = test_data.iloc[:, 1:1+4]
    np_test_data = np.array(test_data)

    # if config.dataset == 'test_2':  # 数据集2联合训练
    #     test_data_rate = 0.3
    #     test_train_data = np_test_data[:int(np_test_data.shape[0] * test_data_rate), :]
    #     np_train_data = np.concatenate([test_train_data, np_train_data], axis=0)

    if config.normalization:
        """z-score 归一化"""
        scaler = StandardScaler()
        np_train_data_normal = scaler.fit_transform(np_train_data)
        np_test_data_normal = scaler.transform(np_test_data)
        """0-1 归一化"""
        # ls_scaler = []
        # for i in range(config.capacity):
        #     scaler = one_zero_normalization(scale_min=0, scale_max=1)
        #     ls_scaler.append(scaler)
        #     if i == 0:
        #         np_train_data_normal = scaler.fit_transform(np_train_data[:, i:i + 1], v_max=None, v_min=None)
        #         np_test_data_normal = scaler.transform(np_test_data[:, i:i + 1])
        #     else:
        #         train_data_normal = scaler.fit_transform(np_train_data[:, i:i + 1], v_max=None, v_min=None)
        #         test_data_normal = scaler.transform(np_test_data[:, i:i + 1])
        #         np_train_data_normal = np.concatenate([np_train_data_normal, train_data_normal], axis=1)
        #         np_test_data_normal = np.concatenate([np_test_data_normal, test_data_normal], axis=1)
        return np_train_data_normal, np_test_data_normal, test_date_list, scaler
    else:
        scaler = None
        return np_train_data, np_test_data, test_date_list, scaler



if __name__ == '__main__':



    parser = argparse.ArgumentParser(description='main')
    parser.add_argument("--dataset", type=str, default="test_1")
    parser.add_argument("--capacity", type=int, default=4)
    # parser.add_argument("--dataset", type=str, default="test_2")
    parser.add_argument("--train_downsample", type=int, default=1)
    parser.add_argument("--test_downsample", type=int, default=1)
    parser.add_argument("--normalization", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--model", type=str, default='SVM')
    # MA, V_AR, ARI_MA, prophet, RF, XGB, Stacking, SVM, TGRNN, ANN, LSTM, Seq2Seq, TF, DLinear
    parser.add_argument("--train", type=bool, default=True)
    parser.add_argument("--direct_predict_multi_step", type=bool, default=False,
                        help='just for some ML method, because some ML method do not support direct multi-step predict')
    parser.add_argument("--input_len", type=int, default=45)
    parser.add_argument("--pred_len", type=int, default=15)
    parser.add_argument("--var_len", type=int, default=1)  # input dim
    parser.add_argument("--out_dim", type=int, default=1)  # output dim

    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--patient", type=int, default=10)
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--save_pred_result", type=bool, default=True, help='save metrics')
    parser.add_argument("--save_pred_data", type=bool, default=True, help='save gt and pred data')

    args = parser.parse_args()
    dict_args = vars(args)
    config = edict(dict_args)
    recent_best_results = deque(maxlen=5)
    for k in range(5):
        n = 20+2*k
        seed= config.seed = 3407
        set_seed(3407)
        c= 0.03+0.005*k





        # read data
        if config.dataset == 'test_1':
            config.train_file = "../TM_data/Scenario_1/TM_train_1min_2023_0727-0731.csv"
            config.test_file = '../TM_data/Scenario_1/TM_test_1min_2024_0403-0413.csv'
        elif config.dataset == 'test_2':
            # config.train_file = "../data/unit2_2401-2403_I+P_shutdown.csv"
            config.train_file = "../TM_data/Scenario_2/train_double_1h_drift_2023_0217-0331.csv"
            # config.train_file = '../data/unit2_2401-2403_I+P_startup3.csv'

            # config.test_file = '../data/unit2_2401-2403_I+P_startup2.csv'
            config.test_file = '../TM_data/Scenario_2/test_1h_2023_0910-20240131.csv'
            # config.test_file = "../data/unit2_2401-2403_I+P_shutdown.csv"
        else:
            raise ValueError(f'Not find dataset: {config.dataset}')
        config.out_path = f'./output/{config.model}/{config.dataset}_{config.input_len}_{k}'
        np_train_data, np_test_data, test_date_list, ls_scaler = get_data(config)
        os.makedirs(config.out_path, exist_ok=True)
        # train and predict
        model = get_model(config,seed,n,c)

        # 统计学方法
        if config.model in ['MA', 'V_AR', 'ARI_MA']:
            model.fit(np_train_data)
            all_pred = model.predict(np_test_data)
            all_gt = np_test_data[config.input_len:, :]  # 去头
            all_gt = all_gt[0:len(all_pred)]  # 去尾
        # ML
        elif config.model in ['RF', 'TGRNN', 'XGB', 'Stacking', 'SVM']:
            train_x, train_y = ML_dataset(np_train_data, input_len=config.input_len, pred_len=config.pred_len, step=1)
            test_x, test_y = ML_dataset(np_test_data, input_len=config.input_len, pred_len=config.pred_len,
                                        step=config.pred_len)
            if config.train:
                model.fit(train_x, train_y,)
            all_pred = model.predict(test_x)
            all_gt = test_y.reshape((-1, test_y.shape[-1]))
        # prophet
        elif config.model in ['prophet']:
            model.fit(np_train_data)
            all_pred = model.predict(np_test_data, test_date_list)
            all_gt = np_test_data[config.input_len:, :]  # 去头
            all_gt = all_gt[0:len(all_pred)]  # 去尾
        # DL
        else:
            train_dataloader = dataset(np_train_data, config.input_len, config.pred_len, step=1, bs=config.batch_size)
            test_dataloader = dataset(np_test_data, config.input_len, config.pred_len, step=config.pred_len, bs=1,
                                      shuffle=False)
            if config.train:
                model.fit(train_dataloader)
            all_pred, all_gt = model.predict(test_dataloader)

        if config.normalization:
            all_pred = ls_scaler.inverse_transform(all_pred)
            all_gt = ls_scaler.inverse_transform(all_gt)
            # all_pred = Anti_testnormalization(all_pred, ls_scaler)
            # all_gt = Anti_testnormalization(all_gt, ls_scaler)

        # plot result
        plot_save_result(config, all_gt, all_pred)

        # show metrics
        all_rmse, all_mae, all_mape, all_smape, all_RAE = evaluate_forecasts(all_pred, all_gt)
        if config.save_pred_result:  # save metrics
            all_rmse_np = np.array(all_rmse).reshape((-1, 1))
            all_mae_np = np.array(all_mae).reshape((-1, 1))
            all_mape_np = np.array(all_mape).reshape((-1, 1))
            all_smape_np = np.array(all_smape).reshape((-1, 1))
            all_RAE_np = np.array(all_RAE).reshape((-1, 1))
            all_metric = np.concatenate([all_rmse_np, all_mae_np, all_mape_np, all_smape_np, all_RAE_np], axis=1)
            all_metric = pd.DataFrame(all_metric, columns=['RMSE', 'MAE', 'MAPE', 'SMAPE', 'RAE'])

            metric_path = f'../save_result/metric/{config.dataset}'
            os.makedirs(metric_path, exist_ok=True)
            all_metric.to_csv(f'{metric_path}/{config.model}_{config.input_len}.csv')

        delta_rmse = np.std(all_rmse[:config.capacity])
        delta_mae = np.std(all_mae[:config.capacity])
        delta_mape = np.std(all_mape[:config.capacity])
        delta_smape = np.std(all_smape[:config.capacity])
        delta_RAE = np.std(all_RAE[:config.capacity])

        rmse = np.mean(all_rmse[:config.capacity])
        mae = np.mean(all_mae[:config.capacity])
        mape = np.mean(all_mape[:config.capacity])
        smape = np.mean(all_smape[:config.capacity])
        RAE = np.mean(all_RAE[:config.capacity])



        print('Model: {},Seed: {}'.format(config.model,config.seed))
        print('filename: {},train_file: {},test_file: {}'.format(config.dataset,config.train_file,config.test_file))
        for i in range(len(all_mae)):


            print('~~The sensor_{} forecasting results: '
                  'RMSE: {:.6f}, MAE: {:.6f}, MAPE:{:.6f}%, SMAPE:{:.6f}% , RAE:{:.6f}'.
                  format(i, all_rmse[i], all_mae[i], all_mape[i], all_smape[i], all_RAE[i]))
        print('--All average SPNDs+Sensors fitting results: '
              'RMSE: {:.6f}, MAE: {:.6f}, MAPE:{:.6f}%, SMAPE:{:.6f}%, RAE:{:.6f}'.
              format(rmse, mae, mape, smape, RAE))
        print('                                        std: '
              'RMSE: {:.6f}, MAE: {:.6f}, MAPE:{:.6f}%, SMAPE:{:.6f}%, RAE:{:.6f}'.
              format(delta_rmse, delta_mae, delta_mape, delta_smape, delta_RAE))

        recent_best_results.append((seed, all_rmse, all_mae, all_mape, all_smape, all_RAE))

    rmse_five_epoch_mean=[]
    mae_five_epoch_mean=[]
    mape_five_epoch_mean=[]
    smape_five_epoch_mean=[]
    RAE_five_epoch_mean=[]
    for i,result in enumerate(recent_best_results):
        print('-{}_seed------------------'.format(seed))
        print('-{}_seed------------------'.format(seed))
        seed, best5_rmse, best5_mae, best5_mape, best5_smape, best5_RAE = result

        rmse = np.mean(best5_rmse[:config.capacity])
        mae = np.mean(best5_mae[:config.capacity])
        mape = np.mean(best5_mape[:config.capacity])
        smape = np.mean(best5_smape[:config.capacity])
        RAE = np.mean(best5_RAE[:config.capacity])

        delta_rmse = np.std(best5_rmse[:config.capacity])
        delta_mae = np.std(best5_mae[:config.capacity])
        delta_mape = np.std(best5_mape[:config.capacity])
        delta_smape = np.std(best5_smape[:config.capacity])
        delta_RAE = np.std(best5_RAE[:config.capacity])

        rmse_five_epoch_mean.append(rmse)
        mae_five_epoch_mean.append(mae)
        mape_five_epoch_mean.append(mape)
        smape_five_epoch_mean.append(smape)
        RAE_five_epoch_mean.append(RAE)

        for j in range(len(all_mae)):
            print('-{}seed~~The sensor_{} forecasting results: '
                  'RMSE: {:.6f}, MAE: {:.6f}, MAPE:{:.6f}%, SMAPE:{:.6f}% , RAE:{:.6f}'.
                  format(i, j, best5_rmse[j], best5_mae[j], best5_mape[j], best5_smape[j], best5_RAE[j]))
        print('-{}seed--All average SPNDs+Sensors fitting results: '
              'RMSE: {:.6f}, MAE: {:.6f}, MAPE:{:.6f}%, SMAPE:{:.6f}%, RAE:{:.6f}'.
              format(i, rmse, mae, mape, smape, RAE))
        print('-{}seed                                        std: '
              'RMSE: {:.6f}, MAE: {:.6f}, MAPE:{:.6f}%, SMAPE:{:.6f}%, RAE:{:.6f}'.
              format(i, delta_rmse, delta_mae, delta_mape, delta_smape, delta_RAE))
    print('----5--seed----Mean--and--std---for---SD---SE----')
    print('----5--seed----Mean--and--std---for---SD---SE----')
    print('----5--seed----Mean--and--std---for---SD---SE----')
    print('----5--seed----Mean--and--std---for---SD---SE----')
    rmse_five_epoch_mean_average = np.mean(rmse_five_epoch_mean)
    mae_five_epoch_mean_average = np.mean(mae_five_epoch_mean)
    mape_five_epoch_mean_average = np.mean(mape_five_epoch_mean)
    smape_five_epoch_mean_average = np.mean(smape_five_epoch_mean)
    RAE_five_epoch_mean_average = np.mean(RAE_five_epoch_mean)

    delta_rmse_five_epoch_mean = np.std(rmse_five_epoch_mean)
    delta_mae_five_epoch_mean = np.std(mae_five_epoch_mean)
    delta_mape_five_epoch_mean = np.std(mape_five_epoch_mean)
    delta_smape_five_epoch_mean = np.std(smape_five_epoch_mean)
    delta_RAE_five_epoch_mean = np.std(RAE_five_epoch_mean)

    print(
        '----rmse_five_epoch_mean_average:{:.6f},----mae_five_epoch_mean_average:{:.6f},----mape_five_epoch_mean_average:{:.6f},----smape_five_epoch_mean_average:{:.6f},----RAE_five_epoch_mean_average:{:.6f}'
        .format(rmse_five_epoch_mean_average, mae_five_epoch_mean_average, mape_five_epoch_mean_average,
                smape_five_epoch_mean_average, RAE_five_epoch_mean_average))
    print(
        '----delta_rmse_five_epoch_mean:{:.6f},----delta_mae_five_epoch_mean:{:.6f},----delta_mape_five_epoch_mean:{:.6f},----delta_smape_five_epoch_mean:{:.6f},----delta_RAE_five_epoch_mean:{:.6f},'
        .format(delta_rmse_five_epoch_mean, delta_mae_five_epoch_mean, delta_mape_five_epoch_mean,
                delta_smape_five_epoch_mean,
                delta_RAE_five_epoch_mean))

    print('--------------Print Ending-----------------')
    print('--------------Print Ending-----------------')
    print('--------------Print Ending-----------------')
    print('--------------Print Ending-----------------')

