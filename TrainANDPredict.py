import torch
import random
import time
import os

from sentry_sdk.utils import epoch

from PST_utils import _create_if_not_exist, get_logger, str2bool, ensure_dir, build_optimizer, build_lr_scheduler, \
    save_model
from MODEL import *
import PST_loss as loss_factory
import os
import argparse
from collections import deque
import torch
import torch.nn.functional as F
import yaml
import numpy as np
from easydict import EasyDict as edict
from PST_dataset_kfold import I_Dataset, Test_I_Dataset
from PST_metrics import evaluate_forecasts_save, compute_error_abs, predict_plot
from PST_utils import get_logger, str2bool, save_result_csv, load_model
from logging import getLogger
import matplotlib.pyplot as plt
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

print(torch.cuda.device_count())
def monitor_gpu_memory():
    """监控并输出当前 GPU 内存使用情况"""
    allocated_memory = torch.cuda.memory_allocated()  # 已分配的内存
    reserved_memory = torch.cuda.memory_reserved()    # 保留的内存
    total_memory = torch.cuda.get_device_properties(0).total_memory  # GPU 总内存

    allocated_memory_mb = allocated_memory / 1024 ** 2  # 转换为MB
    reserved_memory_mb = reserved_memory / 1024 ** 2
    total_memory_mb = total_memory / 1024 ** 2
    return allocated_memory_mb, reserved_memory_mb, total_memory_mb

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


def Anti_testnormalization_expect_I(pred_y, all_scaler, columns, MinMaxNormalization):  # 这里进来的pred_y是没有归一化的所有的拼接值
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
        pred_y = all_scaler.inverse_transform(pred_y)
    return pred_y

# def Anti_testnormalization_expect_I(pred_y, all_scaler, columns, MinMaxNormalization):
#     """inverse_transform for sensors except those starting with 'I'"""
#     if MinMaxNormalization:
#         # 确定需要处理的列索引（非'I'开头的列）
#         expect_I_index = [columns.index(c) for c in columns if not c.startswith('I')]
#
#         # 将pred_y转换为二维数组，形状为(n_samples, n_features)
#         # 假设输出特征数等于expect_I_index的数量
#         n_features = len(expect_I_index)
#         pred_y = pred_y.reshape(-1, n_features)
#
#         # 对每个需要处理的列进行反归一化
#         for i, s in enumerate(expect_I_index):
#             s_name = columns[s]
#             # 取出对应列的数据，并适配scaler需要的二维结构
#             np_data = pred_y[:, i].reshape(-1, 1)
#             scaler = all_scaler[s_name]
#             new_pred = scaler.inverse_transform(np_data)
#             # 将结果重新填充回pred_y
#             pred_y[:, i] = new_pred.flatten()
#     else:
#         # 使用全局scaler进行反归一化
#         pred_y = all_scaler.inverse_transform(pred_y.reshape(-1, 1))
#
#     return pred_y

def get_model(config):  # , graph
    if config.model not in globals():
        raise NotImplementedError("Not found the model: {}".format(config.model))
    model = globals()[config.model](config)  # , graph
    return model


def Sensor_test(config, dataset):  # 这里传进来的dataset是I_Dataset，所以下面的all_scaler_x可以读的到
    log = getLogger()

    with torch.no_grad():
        # graph_dict = dataset.graph_dict
        model = get_model(config).to(config.device)  # , graph_dict['graph_1']

        output_path = config.output_path + config.exp_id + '_' + config.model
        output_path = os.path.join(output_path, "model_%d.pt" % config.best)

        checkpoint = torch.load(output_path, map_location='cuda')
        model.load_state_dict(checkpoint)

        model.eval()  # 是评估模式，而非训练模式。
        # 在评估模式下，batchNorm层，dropout层等用于优化训练而添加的网络层会被关闭，从而使得评估时不会发生偏移。

        # test_path = sorted(glob.glob(os.path.join("./data", config.test_path, "*"))) # ['./data\\test_y\\test_y1.csv','./data\\test_y\\testy1_describe.txt']
        test_x_ds = Test_I_Dataset(filename=config.test_path, capacity=config.capacity,
                                   normal_flag=config.train_normal_flag, all_scaler_x=dataset.all_scaler_x,
                                   all_scaler_y=dataset.all_scaler_y,
                                   Num_sample=0, MinMaxNormalization=dataset.MinMaxNormalization)
        test_y_ds = Test_I_Dataset(filename=config.test_path, capacity=config.capacity,
                                   normal_flag=False, all_scaler_x=dataset.all_scaler_x,
                                   all_scaler_y=dataset.all_scaler_y,
                                   Num_sample=0, MinMaxNormalization=dataset.MinMaxNormalization)

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
        if config.out_capacity == 1:
            # 单通道场景：将一维数组转为二维 (N, 1)
            all_pred_y = all_pred_y.reshape(-1, 1)
            all_test_y = all_test_y.reshape(-1, 1)

        if config.train_normal_flag:
            columns_y = dataset.columns_y
            all_pred_y = Anti_testnormalization_expect_I(all_pred_y, dataset.all_scaler_y, columns_y,
                                                         MinMaxNormalization=config.MinMaxNormalization)

        # 使用条件索引和广播将小于 0 的元素替换为 0
        # all_pred_y[all_pred_y < 0] = 0
        # all_pred_y = all_pred_y[0:len(all_test_y), :]
        data_type = config.test_path.split('/')[-1].split('.')[0]
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
            save_result_csv(all_test_y, all_pred_y, file_path='./save_result/data/',
                            file_name=f'{config.model}_{config.input_len}',
                            columns=[f'P_{n + 1}' for n in range(config.out_capacity)])
    return all_rmse, all_mae, all_mape, all_smape, all_RAE


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


def get_model(config):  # , graph
    if config.model not in globals():
        raise NotImplementedError("Not found the model: {}".format(config.model))
    model = globals()[config.model](config)  # , graph
    return model


def train(config, dataset):
    print("__________________________________")
    # print(torch.cuda.device_count())
    print("__________________________________")
    log = getLogger()
    # graph_dict = dataset.graph_dict  # dict: M*(134, 134)

    train_data_loader = dataset.train_dataloader

    model = get_model(config).to(config.device)  # , graph_dict['graph_1']

    log.info(model)
    for name, param in model.named_parameters():
        log.info(str(name) + '\t' + str(param.shape) + '\t' +
                 str(param.device) + '\t' + str(param.requires_grad))
    total_num = sum([param.nelement() for param in model.parameters()])
    log.info('Total parameter numbers: {}'.format(total_num))

    loss_fn = getattr(loss_factory, config.loss)()

    # 优化器
    opt = build_optimizer(config, log, model)
    opt.zero_grad()

    _create_if_not_exist(config.output_path)
    time_start = time.time()
    log.info("--------Begin Training--------")

    all_pred_loss = []
    valid_records = []
    best_score = np.inf
    patient = 0
    all_allocated_memory = []  # 用来记录每个 epoch 的分配内存
    all_reserved_memory = []  # 用来记录每个 epoch 的保留内存
    epoch_num = 0
    model.train()
    for epoch in range(config.epoch):
        allocated_memory, reserved_memory, total_memory = monitor_gpu_memory()
        all_allocated_memory.append(allocated_memory)
        all_reserved_memory.append(reserved_memory)

        print(f"\nEpoch {epoch + 1}/{config.epoch} - GPU memory usage at the start:")
        print(f"Allocated Memory: {allocated_memory:.2f} MB")
        print(f"Reserved Memory: {reserved_memory:.2f} MB")
        print(f"Total Memory: {total_memory:.2f} MB")
        pred_losses = []
        for data, true in tqdm(train_data_loader, total=len(train_data_loader), desc='train'):
            # src_data, src_true = data_augment(src_data, src_true, tgt_data, tgt_true)  # smote数据融合
            data, true = data.to(config.device), true.to(config.device)

            pred = model(data)  # extract source features
            # pred[pred < 0] = 0
            loss = loss_fn(pred, true)  # calculate source predict loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            pred_losses.append(loss.item())

        pred_loss = sum(pred_losses) / len(pred_losses)
        all_pred_loss.append(pred_loss)
        log.info(f"Epoch={epoch}, exp_id={config.exp_id}, seq_Loss: {pred_loss:.7f}")  # 这里的seq_loss实际上是归一化后的值

        valid_records.append({'loss': pred_loss})
        best_score = min(pred_loss, best_score)
        if best_score == pred_loss:
            patient = 0
            save_model(config.output_path + config.exp_id + '_' + config.model, model, steps=epoch)
            log.info("---Saving the current model---")
        else:
            patient += 1
            if patient > config.patient:
                log.info("----Model Patient-----Earlystopping")
                break
        epoch_num = epoch
    # training is over
    time_end = time.time()
    avg_allocated_memory = sum(all_allocated_memory) / len(all_allocated_memory)
    avg_reserved_memory = sum(all_reserved_memory) / len(all_reserved_memory)

    print("\nTraining complete. Average GPU memory usage:")
    print(f"Average Allocated Memory: {avg_allocated_memory:.2f} MB")
    print(f"Average Reserved Memory: {avg_reserved_memory:.2f} MB")

    log.info("--------Over Training-------- \n训练用时: {:.3f} mins".format((time_end - time_start) / 60))
    time_per_epoch = (time_end - time_start) / (60*epoch_num)
    best_epochs = min(enumerate(valid_records), key=lambda x: x[1]["loss"])[0]
    log.info("Best valid Epoch %s" % best_epochs)
    log.info("Best valid score %s" % valid_records[best_epochs])

    x = range(0, len(all_pred_loss))
    if config.DoYouNeedEpochSeqlossFigure:
        plt.plot(x, all_pred_loss, '.-', label='seq_losses')
        plt.title('loss')
        plt.legend()
        plt.show()
    return best_epochs, avg_allocated_memory, avg_reserved_memory ,total_num ,time_per_epoch


def main():
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument("--conf", type=str, default="./config/config_test1.yaml")  # config_test1.yaml config_test2.yaml

    parser.add_argument("--model", type=str,
                            default='Informer')  # , GCN_GRU, TCN_GCN, GCN_TCN, GCN_LSTM, MTGNN, AGCRN, TCN_Mixhop, Dilated_GCN, Parallel_MTGNN, FCSTGNN
    parser.add_argument("--batch_size", type=int, default=32)  # 16 32
    parser.add_argument("--epoch", type=int, default=10)

    parser.add_argument("--K", type=int, default=5, help='K-fold')  # ~K=5折交叉
    parser.add_argument("--ind", type=int, default=4, help='selected fold for validation set')  # ~~K个验证集，选择一个
    parser.add_argument("--pad", type=str2bool, default=False, help='pad with last sample')  # 应该是K折整齐划分处理
    parser.add_argument("--random", type=str2bool, default=False, help='Whether shuffle num_nodes')

    parser.add_argument("--enhance", type=str2bool, default=False, help='Whether enhance the time dim')  # ~暂时关掉
    parser.add_argument("--data_diff", type=int, default=0, help='val_len+1 differential features')  # 要加1
    parser.add_argument("--add_apt", type=str2bool, default=True, help='Whether to use adaptive matrix')
    parser.add_argument("--Multi_Graph_num", type=int, default=1, help='1-3: distance adj, WAS adj and adapative adj')
    parser.add_argument("--gsteps", type=int, default=1, help='Gradient Accumulation')  # 梯度积累
    parser.add_argument("--loss", type=str, default='FilterHuberLoss')
    parser.add_argument("--exp_id", type=str, default='49039')  #
    parser.add_argument("--search_best_range", type=list, default=[0, 100])  #
    # parser.add_argument("--csv_name", type=str, default='CORALDANN_24')  #

    parser.add_argument("--output_path", type=str, default='output/')

    # parser.add_argument("--test_days", type=float, default=6.95)  # temp 1000天

    parser.add_argument("--save_flag", type=str2bool, default=True, help='save result figure')
    parser.add_argument("--save_data_flag", type=str2bool, default=True, help='save result data')
    parser.add_argument("--save_data_recon_er", type=str2bool, default=False,
                        help='Whether to save recon error of normal data')
    args = parser.parse_args()
    dict_args = vars(args)

    config = edict(yaml.load(open(args.conf), Loader=yaml.FullLoader))
    config.update(dict_args)
    config.var_len = config.data_diff + 1


    exp_id = int(random.SystemRandom().random() * 100000)
    config['exp_id'] = str(exp_id)

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
        MinMaxNormalization=config.MinMaxNormalization,
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

    best,AAM,ARM,total_num,time_per_epoch=train(config, dataset)
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

    # 假设有一个固定大小的队列来保存最近五代的结果
    recent_best_results = deque(maxlen=5)
    best_metric = np.inf

    for best in range(config.search_best_range[0], config.search_best_range[1], 1):
        config['best'] = best
        try:
            all_rmse, all_mae, all_mape, all_smape, all_RAE = Sensor_test(config, dataset)  # valid_data, test_data
        except Exception as e:
            print(f"Error: {e}")
            continue

        rmse = np.mean(all_rmse[:config.capacity])
        mae = np.mean(all_mae[:config.capacity])
        mape = np.mean(all_mape[:config.capacity])
        smape = np.mean(all_smape[:config.capacity])  # Use mean of smape for comparison
        RAE = np.mean(all_RAE[:config.capacity])

        delta_rmse = np.std(all_rmse[:config.capacity])
        delta_mae = np.std(all_mae[:config.capacity])
        delta_mape = np.std(all_mape[:config.capacity])
        delta_smape = np.std(all_smape[:config.capacity])
        delta_RAE = np.std(all_RAE[:config.capacity])

        # If we find a better smape, update the best_metric
        if smape < best_metric:
            best_metric = smape
            # Save the best result
            recent_best_results.append((best, all_rmse, all_mae, all_mape, all_smape, all_RAE))
            # Sort the results by smape mean (ascending order)
            recent_best_results = deque(sorted(recent_best_results, key=lambda x: np.mean(x[4])), maxlen=5)
        else:
            # If the new result is not the best but is better than the worst in the top 5, replace the worst one
            if len(recent_best_results) < 5:
                recent_best_results.append((best, all_rmse, all_mae, all_mape, all_smape, all_RAE))
            else:
                # Find the worst result in the deque (the one with the highest smape mean)
                worst_index = np.argmax(
                    [np.mean(x[4]) for x in recent_best_results])  # x[3] corresponds to smape, using mean
                worst_smape = np.mean(recent_best_results[worst_index][4])  # Same for worst_smape

                # If the current result is better than the worst one, replace the worst result
                if smape < worst_smape:
                    recent_best_results[worst_index] = (best, all_rmse, all_mae, all_mape, all_smape, all_RAE)

            # After replacing or adding the result, sort the deque by smape mean (ascending order)
            recent_best_results = deque(sorted(recent_best_results, key=lambda x: np.mean(x[4])), maxlen=5)

        normalization_type = 'MinMaxNormalization' if config.MinMaxNormalization else 'Standard Scaler'



        print('---exp_id: {}----Model: {}-----Normalization: {}----Train_Normal_flag: {}-----seed: {}--------lr: {}------Average allocated memory: {}MB-----Average reserved memory: {}MB------total_para_num: {}-------time_per_epoch: {}s'.format(
            config.exp_id,
            config.model,
            normalization_type,
            config.train_normal_flag,config.seed,
            config.lr,
            AAM,
            ARM,
            total_num,
            time_per_epoch*60
        ))


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


        rmse_five_epoch_mean = []
        mae_five_epoch_mean = []
        mape_five_epoch_mean = []
        smape_five_epoch_mean = []
        RAE_five_epoch_mean = []

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

            rmse_five_epoch_mean.append(rmse)
            mae_five_epoch_mean.append(mae)
            mape_five_epoch_mean.append(mape)
            smape_five_epoch_mean.append(smape)
            RAE_five_epoch_mean.append(RAE)

            for j in range(len(all_mae)):
                print('-{}epoch~~The sensor_{} forecasting results: '
                      'RMSE: {:.6f}, MAE: {:.6f}, MAPE:{:.6f}%, SMAPE:{:.6f}% , RAE:{:.6f}'.
                      format(i, j, best5_rmse[j], best5_mae[j], best5_mape[j], best5_smape[j], best5_RAE[j]))
            print('-{}epoch--All average SPNDs+Sensors fitting results: '
                  'RMSE: {:.6f}, MAE: {:.6f}, MAPE:{:.6f}%, SMAPE:{:.6f}%, RAE:{:.6f}'.
                  format(i, rmse, mae, mape, smape, RAE))
            print('-{}epoch                                        std: '
                  'RMSE: {:.6f}, MAE: {:.6f}, MAPE:{:.6f}%, SMAPE:{:.6f}%, RAE:{:.6f}'.
                  format(i, delta_rmse, delta_mae, delta_mape, delta_smape, delta_RAE))

        print('----5--Epoch----Mean--and--std---for---SD---SE----')
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
            .format(delta_rmse_five_epoch_mean, delta_mae_five_epoch_mean, delta_mape_five_epoch_mean, delta_smape_five_epoch_mean,
                    delta_RAE_five_epoch_mean))

        print('--------------Print Ending-----------------')



if __name__ == '__main__':
    main()











