import argparse
import yaml
import torch
import random
import numpy as np
import time
import pandas as pd
import os
from logging import getLogger
from easydict import EasyDict as edict
from matplotlib import pyplot as plt
from tqdm import tqdm
from PST_dataset_kfold import I_Dataset
from PST_utils import _create_if_not_exist, get_logger, str2bool, ensure_dir, build_optimizer, build_lr_scheduler, \
    save_model
from MODEL import *
import PST_loss as loss_factory
from PST_train import train
from PST_predict import Sensor_test

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
print(torch.cuda.device_count())


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


def get_model(config, graph):
    if config.model not in globals():
        raise NotImplementedError("Not found the model: {}".format(config.model))
    model = globals()[config.model](config, graph)
    return model

def read_config():
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument("--conf", type=str, default="../config/config_test2.yaml")  # config_test1.yaml config_test2.yaml

    parser.add_argument("--model", type=str,
                        default='AGCRN')
    # GCN_GRU, TCN_GCN, GCN_TCN, GCN_LSTM, MTGNN, AGCRN, TCN_Mixhop, Dilated_GCN, Parallel_MTGNN, FCSTGNN
    parser.add_argument("--batch_size", type=int, default=32)  # 16 32
    parser.add_argument("--epoch", type=int, default=100)

    parser.add_argument("--K", type=int, default=5, help='K-fold')  # ~K=5折交叉
    parser.add_argument("--ind", type=int, default=4, help='selected fold for validation set')  # ~~K个验证集，选择一个
    parser.add_argument("--pad", type=str2bool, default=False, help='pad with last sample')  # 应该是K折整齐划分处理
    parser.add_argument("--random", type=str2bool, default=False, help='Whether shuffle num_nodes')

    parser.add_argument("--enhance", type=str2bool, default=False, help='Whether enhance the time dim')  # ~暂时关掉
    parser.add_argument("--data_diff", type=int, default=1, help='val_len+1 differential features')  # 要加1
    parser.add_argument("--add_apt", type=str2bool, default=True, help='Whether to use adaptive matrix')
    parser.add_argument("--gcn_k", type=int, default=3)
    parser.add_argument("--dilation_0", type=int, default=4)
    parser.add_argument("--graph_file", type=list, default=['48_node_train_graph_6.xlsx'])
    parser.add_argument("--gpu_id: 0", type=int, default=0)
    parser.add_argument("--Multi_Graph_num", type=int, default=1, help='1-3: distance adj, WAS adj and adapative adj')
    parser.add_argument("--gsteps", type=int, default=1, help='Gradient Accumulation')  # 梯度积累
    parser.add_argument("--loss", type=str, default='FilterHuberLoss')
    parser.add_argument("--save_flag", type=str2bool, default=False, help='save result figure')
    parser.add_argument("--save_data_flag", type=str2bool, default=False, help='save result data')
    parser.add_argument("--save_data_recon_er", type=str2bool, default=False,
                        help='Whether to save recon error of normal data')

    args = parser.parse_args()
    dict_args = vars(args)
    config = edict(yaml.load(open(args.conf), Loader=yaml.FullLoader))
    config.update(dict_args)
    config.var_len = config.data_diff + 1

    config.data_path = "../data/"
    config.graph_path = "../Gene_graph/Graph_data1"
    if config.conf == "../config/config_test1.yaml":
        config.test_path = '../data/unit2_2307-2312_I+P_part123.csv'
    elif config.conf == "../config/config_test2.yaml":
        config.test_path = '../data/unit2_2401-2403_I+P_startup3.csv'
    return config


def main(config, dict_para):
    config.update(dict_para)
    config.var_len = config.data_diff + 1

    exp_id = int(random.SystemRandom().random() * 100000)
    config['exp_id'] = str(exp_id)
    config['output_path'] = './output/OP/'

    logger = get_logger(config)
    logger.info(config)
    set_seed(config.seed)
    ensure_dir(config.output_path)

    dataset = I_Dataset(
        data_path=config.data_path,
        filename=config.filename,
        capacity=config.capacity,
        batch_size=config.batch_size,
        weight_adj_epsilon=config.weight_adj_epsilon,
        K=config.K,
        ind=config.ind,
        Multi_Graph_num=config.Multi_Graph_num,
        binary=config.binary,
        train_normal_flag=config.train_normal_flag,
        test_normal_flag=config.test_normal_flag,
        num_workers=config.num_workers,
        pad_with_last_sample=config.pad,
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
    best_id = train(config, dataset)

    config['best'] = best_id
    best_metric = np.inf
    for best in range(0, 150):
        config['best'] = best
        try:
            all_rmse, all_mae, all_mape, all_smape, all_RAE = Sensor_test(config, dataset)
        except:
            continue
        rmse = np.mean(all_rmse[:config.capacity])
        mae = np.mean(all_mae[:config.capacity])
        mape = np.mean(all_mape[:config.capacity])
        smape = np.mean(all_smape[:config.capacity])
        RAE = np.mean(all_RAE[:config.capacity])

        if smape < best_metric:
            best_metric = smape
            text = f"{config.exp_id}/{best},{dict_para}~~\n" + 20 * " " + 'RMSE: {:.6f}, MAE: {:.6f}, MAPE:{:.6f}%, SMAPE:{:.6f}%, RAE:{:.6f}\n'.format(
        rmse, mae, mape, smape, RAE)
            result_list = [rmse, mae, mape, smape, RAE]
    return text, result_list


if __name__ == '__main__':
    config = read_config()
    para = {'gcn_k': [2, 3], 'graph_file': [4, 6, 8, 10], 'lr': [0.001, 0.002, 0.003]}   # 2
    for gcn_k in para['gcn_k']:
        for graph in para['graph_file']:
            for l in para['lr']:
                para_dict = {'gcn_k': gcn_k, 'graph_file': [f"train_graph_{graph}.xlsx"], 'lr': l}
                result_str, result_list = main(config, para_dict)
                with open(f'../save_result/OP/op_result_test1_compare-{config.model}_{config.input_len}.txt', 'a+') as file:
                    file.write(result_str + '\n')

    #
    # para = {'lr': [0.001, 0.0015, 0.002, 0.003, 0.004, 0.005]}  # 1
    # # para = {'lr': [0.005]}  # 1
    # for lr in para['lr']:
    #     para_dict = {'lr': lr}
    #     result_str, result_list = main(para_dict)
    #     with open(f'./save_result/OP/op_result_test2_lr_DA_20_lr.txt', 'a+') as file:
    #         file.write(result_str + '\n')
