import os
import glob
import argparse
from audioop import minmax

import torch
import torch.nn.functional as F
import yaml
import numpy as np
import pandas as pd
from easydict import EasyDict as edict
from PST_dataset_kfold import I_Dataset, Test_I_Dataset
from PST_metrics import evaluate_forecasts_save, compute_error_abs, predict_plot
from PST_utils import get_logger, str2bool, save_result_csv, load_model
from logging import getLogger
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from MODEL import *

print(torch.cuda.device_count())


def Anti_testnormalization(pred_y, all_scaler):
    '''
    :param np_data: (N,f)
    :return: (N,f)
    '''
    for i in range(pred_y.shape[1]):
        _scaler = all_scaler[i]
        temp_pred_np = pred_y[:, i].reshape((-1, 1))
        pred_np_original = _scaler.inverse_transform(temp_pred_np)
        if i == 0:
            all_np_pred = pred_np_original
        else:
            all_np_pred = np.concatenate((all_np_pred, pred_np_original), axis=1)
    return all_np_pred


def Anti_testnormalization_expect_I(pred_y, all_scaler, columns,MinMaxNormalization):#这里进来的pred_y是没有归一化的所有的拼接值
    """inverse_transform for other sensors expect I"""
    if MinMaxNormalization:
        expect_I_index = [columns.index(c) for c in columns if c[0] != 'I']
        for s in expect_I_index:
            s_name = columns[s]
            np_data = pred_y[:, s]
            scaler = all_scaler[s_name]
            new_pred = scaler.inverse_transform(np_data)
            pred_y[:, s] = new_pred
    else:
        pred_y=all_scaler.inverse_transform(pred_y)
    return pred_y

def get_model(config):#, graph
    if config.model not in globals():
        raise NotImplementedError("Not found the model: {}".format(config.model))
    model = globals()[config.model](config)#, graph
    return model

def Sensor_test(config, dataset):#这里传进来的dataset是I_Dataset，所以下面的all_scaler_x可以读的到
    log = getLogger()

    with torch.no_grad():
        #graph_dict = dataset.graph_dict
        model = get_model(config).to(config.device)#, graph_dict['graph_1']

        output_path = config.output_path + config.exp_id + '_' + config.model
        output_path = os.path.join(output_path, "model_%d.pt" % config.best)


        checkpoint = torch.load(output_path, map_location='cuda')
        model.load_state_dict(checkpoint)

        model.eval()  # 是评估模式，而非训练模式。
        # 在评估模式下，batchNorm层，dropout层等用于优化训练而添加的网络层会被关闭，从而使得评估时不会发生偏移。

        # test_path = sorted(glob.glob(os.path.join("./data", config.test_path, "*"))) # ['./data\\test_y\\test_y1.csv','./data\\test_y\\testy1_describe.txt']
        test_x_ds = Test_I_Dataset(filename=config.test_path, capacity=config.capacity,
                                    normal_flag=config.train_normal_flag, all_scaler_x=dataset.all_scaler_x, all_scaler_y=dataset.all_scaler_y,
                                    Num_sample=0,MinMaxNormalization = dataset.MinMaxNormalization)
        test_y_ds = Test_I_Dataset(filename=config.test_path, capacity=config.capacity,
                                    normal_flag=False, all_scaler_x=dataset.all_scaler_x, all_scaler_y=dataset.all_scaler_y,
                                    Num_sample=0,MinMaxNormalization = dataset.MinMaxNormalization)

        num_T = test_y_ds.get_data_x().shape[1]
        for i in tqdm(range(0, num_T - config.input_len, config.output_len)):
            if (i + config.input_len + config.output_len) > num_T:  # 避免后面部分数据只有预测值而没有真实值
                break
            test_x = torch.FloatTensor(
                test_x_ds.get_data_x()[:, i:i + config.input_len, :]).to(config.device)  # (B,N,F)
            test_y = torch.FloatTensor(
                test_y_ds.get_data_y()[:, i + config.input_len: i + config.input_len + config.output_len, :]).to(
                config.device)  # [1,6,7,13]
            # test_x (B,N,T,F),[1,44,5,3]
            test_x = test_x.to(config.device)
            pred_y = model(test_x)  # (B,N,T) (1, 1, 36)

            pred_y = pred_y.cpu().numpy().squeeze()  # (1, 20, 24) -> (20, 24)
            test_y = test_y.cpu().numpy().squeeze()

            if i == 0:
                all_pred_y = pred_y
                all_test_y = test_y
            else:
                # (N,B,T,1) np (44,1,7*num_days,1) (44,1,2,1)
                all_pred_y = np.concatenate((all_pred_y, pred_y), axis=0)
                all_test_y = np.concatenate((all_test_y, test_y), axis=0)

        if config.train_normal_flag:
            columns_y = dataset.columns_y
            all_pred_y = Anti_testnormalization_expect_I(all_pred_y, dataset.all_scaler_y, columns_y,MinMaxNormalization=config.MinMaxNormalization)


        # 使用条件索引和广播将小于 0 的元素替换为 0
        # all_pred_y[all_pred_y < 0] = 0
        # all_pred_y = all_pred_y[0:len(all_test_y), :]
        data_type = config.test_path.split('_')[-1].split('.')[0]
        plot_name = [f'{data_type}_I_{j}' for j in range(1, config.capacity + 1)]
        all_rmse, all_mae, all_mape, all_smape, all_RAE = evaluate_forecasts_save(
            all_pred_y, all_test_y, plot_name, config)  # (44,1,1003,1), (44,1,1003,1)

        # look the weight_adapative_max
        # if config.add_apt:  # (55, 55)
        #     L1_DAGG_gate, L1_DAGG_update = model.get_DAGG()
        #     np_L1_DAGG_gate, np_L1_DAGG_update = L1_DAGG_gate.cpu().numpy(), L1_DAGG_update.cpu().numpy()
        #     print("add_apt is True")
        #     print('L1_DAGG_gate:', np_L1_DAGG_gate, '\nL1_DAGG_update:', np_L1_DAGG_update)
        # else:
        #     print("add_apt is False")

        if config.save_data_flag:
            save_result_csv(all_test_y, all_pred_y, file_path='./save_result/data',
                            file_name=f'{config.model}_{config.input_len}', columns=[f'Dpax_{n+1}' for n in range(config.out_capacity)])
    return all_rmse, all_mae, all_mape, all_smape, all_RAE


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument("--conf", type=str, default="./config/config_test1.yaml")  # config_test1.yaml, config_test2.yaml
    parser.add_argument("--model", type=str, default='Informer')
    # GCN_GRU, TCN_GCN, GCN_TCN, GCN_LSTM, MTGNN, AGCRN, TCN_Mixhop, Dilated_GCN, Parallel_MTGNN, FCSTGNN
    parser.add_argument("--exp_id", type=str, default='30594')  #
    parser.add_argument("--search_best_range", type=list, default=[0, 100])  #
    # parser.add_argument("--csv_name", type=str, default='CORALDANN_24')  #

    # parser.add_argument("--input_len", type=int, default=1)  #
    # parser.add_argument("--output_len", type=int, default=30)  #
    # parser.add_argument("--gcn_k", type=int, default=1)
    # parser.add_argument("--dilation_0", type=int, default=1)
    # parser.add_argument("--graph_file", type=list, default=['48_node_train_graph_4.xlsx'])

    parser.add_argument("--output_path", type=str, default='output/')
    parser.add_argument("--K", type=int, default=5, help='K-fold')  # K=5折交叉
    parser.add_argument("--ind", type=int, default=1, help='selected fold for validation set')
    parser.add_argument("--random", type=str2bool, default=False, help='Whether shuffle num_nodes')
    # parser.add_argument("--test_days", type=float, default=6.95)  # temp 1000天

    parser.add_argument("--data_diff", type=int, default=0, help='Whether to use data differential features')
    parser.add_argument("--add_apt", type=str2bool, default=False, help='Whether to use adaptive matrix')  # 3.不同
    parser.add_argument("--Multi_Graph_num", type=int, default=1, help='1-3: distance adj, WAS adj and adapative adj')
    parser.add_argument("--gsteps", type=int, default=1, help='Gradient Accumulation')
    parser.add_argument("--loss", type=str, default='FilterHuberLoss')
    parser.add_argument("--save_flag", type=str2bool, default=True, help='save result figure')
    parser.add_argument("--save_data_flag", type=str2bool, default=False, help='save result data')
    parser.add_argument("--save_data_recon_er", type=str2bool, default=True,
                        help='Whether to save recon error of normal data')
    # parser

    args = parser.parse_args()
    dict_args = vars(args)

    config = edict(yaml.load(open(args.conf), Loader=yaml.FullLoader))
    config.update(dict_args)
    config.var_len = config.data_diff + 1


    logger = get_logger(config)
    logger.info(config)

    size = [config.input_len, config.output_len]  # [144,288]
    dataset = I_Dataset(
        data_path=config.data_path,
        filename=config.filename,
        capacity=config.capacity,
        batch_size=config.batch_size,
        weight_adj_epsilon=config.weight_adj_epsilon,
        MinMaxNormalization=config.MinMaxNormalization,
        K=config.K,
        ind=config.ind,
        Multi_Graph_num=config.Multi_Graph_num,
        binary=config.binary,
        train_normal_flag=config.train_normal_flag,
        test_normal_flag=config.test_normal_flag,
        num_workers=config.num_workers,
        pad_with_last_sample=False,
        size=[config.input_len, config.output_len],
        random=config.random,
        graph_path=config.graph_path,
        graph_file=config.graph_file,
    )

    gpu_id = config.gpu_id
    if gpu_id != -1:
        device = torch.device('cuda:{}'.format(gpu_id))
    else:
        device = torch.device('cpu')
    config['device'] = device

    # # search best from range [config.search_best_range]
    # best_metric = np.inf
    # for best in range(config.search_best_range[0], config.search_best_range[1], 1):
    #     config['best'] = best
    #     try:
    #         all_rmse, all_mae, all_mape, all_smape, all_RAE = Sensor_test(config, dataset)  # valid_data, test_data
    #     except:
    #         continue
    #     rmse = np.mean(all_rmse[:config.capacity])
    #     mae = np.mean(all_mae[:config.capacity])
    #     mape = np.mean(all_mape[:config.capacity])
    #     smape = np.mean(all_smape[:config.capacity])
    #     RAE = np.mean(all_RAE[:config.capacity])
    #
    #     delta_rmse = np.std(all_rmse[:config.capacity])
    #     delta_mae = np.std(all_mae[:config.capacity])
    #     delta_mape = np.std(all_mape[:config.capacity])
    #     delta_smape = np.std(all_smape[:config.capacity])
    #     delta_RAE = np.std(all_RAE[:config.capacity])
    #     print(str(best) + '--All average SPNDs+Sensors fitting results: '
    #           'RMSE: {:.6f}, MAE: {:.6f}, MAPE:{:.6f}%, SMAPE:{:.6f}%, RAE:{:.6f}'.
    #           format(rmse, mae, mape, smape, RAE))
    #     if smape < best_metric:
    #         best_metric = smape
    #         for i in range(len(all_mae)):
    #             print('~~The sensor_{} forecasting results: '
    #                   'RMSE: {:.6f}, MAE: {:.6f}, MAPE:{:.6f}%, SMAPE:{:.6f}% , RAE:{:.6f}'.
    #                   format(i, all_rmse[i], all_mae[i], all_mape[i], all_smape[i], all_RAE[i]))
    #         print('--All average SPNDs+Sensors fitting results: '
    #               'RMSE: {:.6f}, MAE: {:.6f}, MAPE:{:.6f}%, SMAPE:{:.6f}%, RAE:{:.6f}'.
    #               format(rmse, mae, mape, smape, RAE))
    #         print('                                        std: '
    #               'RMSE: {:.6f}, MAE: {:.6f}, MAPE:{:.6f}%, SMAPE:{:.6f}%, RAE:{:.6f}'.
    #               format(delta_rmse, delta_mae, delta_mape, delta_smape, delta_RAE))
    #         print(f'Find the best epoch: {best} !')
    import numpy as np
    from collections import deque

    # 假设有一个固定大小的队列来保存最近五代的结果
    recent_best_results = deque(maxlen=5)
    best_metric = np.inf

    for best in range(config.search_best_range[0], config.search_best_range[1], 1):
        config['best'] = best
        try:
            all_rmse, all_mae, all_mape, all_smape, all_RAE = Sensor_test(config, dataset)  # valid_data, test_data
        except:
            continue

        rmse = np.mean(all_rmse[:config.capacity])
        mae = np.mean(all_mae[:config.capacity])
        mape = np.mean(all_mape[:config.capacity])
        smape = np.mean(all_smape[:config.capacity])
        RAE = np.mean(all_RAE[:config.capacity])

        delta_rmse = np.std(all_rmse[:config.capacity])
        delta_mae = np.std(all_mae[:config.capacity])
        delta_mape = np.std(all_mape[:config.capacity])
        delta_smape = np.std(all_smape[:config.capacity])
        delta_RAE = np.std(all_RAE[:config.capacity])

        print(str(best) + '--All average SPNDs+Sensors fitting results: '
              'RMSE: {:.6f}, MAE: {:.6f}, MAPE:{:.6f}%, SMAPE:{:.6f}%, RAE:{:.6f}'.format(rmse, mae, mape, smape, RAE))

        if smape < best_metric:
            best_metric = smape



            normalization_type = 'MinMaxNormalization' if config.MinMaxNormalization else 'Standard Scaler'

            print('---exp_id: {}----Model: {}-----Normalization: {}----Train_Normal_flag: {}'.format(
                config.exp_id,
                config.model,
                normalization_type,
                config.train_normal_flag
            ))



            print('---exp_id: {}----Model: {}-----Normalization: {}----Train_Normal_flag: {}'.format(
                config.exp_id,
                config.model,
                normalization_type,
                config.train_normal_flag
            ))


            print('---exp_id: {}----Model: {}-----Normalization: {}----Train_Normal_flag: {}'.format(
                config.exp_id,
                config.model,
                normalization_type,
                config.train_normal_flag
            ))


            print('---exp_id: {}----Model: {}-----Normalization: {}----Train_Normal_flag: {}'.format(
                config.exp_id,
                config.model,
                normalization_type,
                config.train_normal_flag
            ))
            print('train_file: {},test_file: {}'.format(config.filename,config.test_path))
            print('train_file: {},test_file: {}'.format(config.filename, config.test_path))
            print('train_file: {},test_file: {}'.format(config.filename, config.test_path))
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
            print(f'Find the best epoch: {best} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')




            # 保存当前结果到最近最佳结果队列
            recent_best_results.append((best, all_rmse, all_mae, all_mape, all_smape, all_RAE))

            rmse_5epoch_mean = []
            mae_5epoch_mean = []
            mape_5epoch_mean = []
            smape_5epoch_mean = []
            RAE_5epoch_mean = []
            delta_rmse_5epoch_mean = []
            delta_mae_5epoch_mean = []
            delta_mape_5epoch_mean = []
            delta_smape_5epoch_mean = []
            delta_RAE_5epoch_mean = []
            # 打印最近五代的结果
            print("~~~~~~~~~~~~~~~~~~~~~Recent 5 best results (including current best):")
            for i, result in enumerate(recent_best_results):
                print('-{}_best_epoch------------------'.format(i))
                print('-{}_best_epoch------------------'.format(i))
                best_epoch, best5_rmse, best5_mae, best5_mape, best5_smape, best5_RAE = result

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

                rmse_5epoch_mean.append(rmse)
                mae_5epoch_mean.append(mae)
                mape_5epoch_mean.append(mape)
                smape_5epoch_mean.append(smape)
                RAE_5epoch_mean.append(RAE)



                for j in range(len(all_mae)):
                    print('-{}epoch~~The sensor_{} forecasting results: '
                          'RMSE: {:.6f}, MAE: {:.6f}, MAPE:{:.6f}%, SMAPE:{:.6f}% , RAE:{:.6f}'.
                          format(i,j, best5_rmse[j], best5_mae[j], best5_mape[j], best5_smape[j], best5_RAE[j]))
                print('-{}epoch--All average SPNDs+Sensors fitting results: '
                      'RMSE: {:.6f}, MAE: {:.6f}, MAPE:{:.6f}%, SMAPE:{:.6f}%, RAE:{:.6f}'.
                      format(i,rmse, mae, mape, smape, RAE))
                print('-{}epoch                                        std: '
                      'RMSE: {:.6f}, MAE: {:.6f}, MAPE:{:.6f}%, SMAPE:{:.6f}%, RAE:{:.6f}'.
                      format(i,delta_rmse, delta_mae, delta_mape, delta_smape, delta_RAE))
            print('----5--Epoch----Mean--and--std---for---SD---SE----')
            print('----5--Epoch----Mean--and--std---for---SD---SE----')
            print('----5--Epoch----Mean--and--std---for---SD---SE----')
            print('----5--Epoch----Mean--and--std---for---SD---SE----')
            rmse_5epoch_mean_average = np.mean(rmse_5epoch_mean)
            mae_5epoch_mean_average = np.mean(mae_5epoch_mean)
            mape_5epoch_mean_average = np.mean(mape_5epoch_mean)
            smape_5epoch_mean_average = np.mean(smape_5epoch_mean)
            RAE_5epoch_mean_average = np.mean(RAE_5epoch_mean)

            delta_rmse_5epoch_mean = np.std(rmse_5epoch_mean)
            delta_mae_5epoch_mean = np.std(mae_5epoch_mean)
            delta_mape_5epoch_mean = np.std(mape_5epoch_mean)
            delta_smape_5epoch_mean = np.std(smape_5epoch_mean)
            delta_RAE_5epoch_mean = np.std(RAE_5epoch_mean)

            print('----rmse_5epoch_mean_average:{:.6f},----mae_5epoch_mean_average:{:.6f},----mape_5epoch_mean_average:{:.6f},----smape_5epoch_mean_average:{:.6f},----RAE_5epoch_mean_average:{:.6f}'
                  .format(rmse_5epoch_mean_average,mae_5epoch_mean_average,mape_5epoch_mean_average,smape_5epoch_mean_average,RAE_5epoch_mean_average))
            print('----delta_rmse_5epoch_mean:{:.6f},----delta_mae_5epoch_mean:{:.6f},----delta_mape_5epoch_mean:{:.6f},----delta_smape_5epoch_mean:{:.6f},----delta_RAE_5epoch_mean:{:.6f},'
                  .format(delta_rmse_5epoch_mean,delta_mae_5epoch_mean,delta_mape_5epoch_mean,delta_smape_5epoch_mean,delta_RAE_5epoch_mean))


            print('--------------Print Ending-----------------')
            print('--------------Print Ending-----------------')
            print('--------------Print Ending-----------------')
            print('--------------Print Ending-----------------')
