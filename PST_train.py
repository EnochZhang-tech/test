import torch
import random
import time
import os
from PST_utils import _create_if_not_exist, get_logger, str2bool, ensure_dir, build_optimizer, build_lr_scheduler, save_model
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


def get_model(config): #, graph
    if config.model not in globals():
        raise NotImplementedError("Not found the model: {}".format(config.model))
    model = globals()[config.model](config) #, graph
    return model

def train(config, dataset):
    print("__________________________________")
    # print(torch.cuda.device_count())
    print("__________________________________")
    log = getLogger()
    #graph_dict = dataset.graph_dict  # dict: M*(134, 134)

    train_data_loader = dataset.train_dataloader

    model = get_model(config).to(config.device) #, graph_dict['graph_1']

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

    model.train()
    for epoch in range(config.epoch):
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
        log.info(f"Epoch={epoch}, exp_id={config.exp_id}, seq_Loss: {pred_loss:.7f}")#这里的seq_loss实际上是归一化后的值

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

    # training is over
    time_end = time.time()
    log.info("--------Over Training-------- \n训练用时: {:.3f} mins".format((time_end - time_start) / 60))
    best_epochs = min(enumerate(valid_records), key=lambda x: x[1]["loss"])[0]
    log.info("Best valid Epoch %s" % best_epochs)
    log.info("Best valid score %s" % valid_records[best_epochs])

    x = range(0, len(all_pred_loss))
    if config.DoYouNeedEpochSeqlossFigure:
        plt.plot(x, all_pred_loss, '.-', label='seq_losses')
        plt.title('loss')
        plt.legend()
        plt.show()
    return best_epochs
def main():
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument("--conf", type=str, default="./config/config_test1.yaml")  # config_test1.yaml config_test2.yaml

    parser.add_argument("--model", type=str, default='Transformer')  # , GCN_GRU, TCN_GCN, GCN_TCN, GCN_LSTM, MTGNN, AGCRN, TCN_Mixhop, Dilated_GCN, Parallel_MTGNN, FCSTGNN
    parser.add_argument("--batch_size", type=int, default=32)  # 16 32
    parser.add_argument("--epoch", type=int, default=300)

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

    parser.add_argument("--search_best_range", type=list, default=[0, 300])  #
    # parser.add_argument("--csv_name", type=str, default='CORALDANN_24')  #

    parser.add_argument("--output_path", type=str, default='output/')

    parser.add_argument("--save_flag", type=str2bool, default=True, help='save result figure')
    parser.add_argument("--save_data_flag", type=str2bool, default=True, help='save result data')
    parser.add_argument("--save_data_recon_er", type=str2bool, default=False,
                        help='Whether to save recon error of normal data')
    args = parser.parse_args()
    dict_args = vars(args)

    config = edict(yaml.load(open(args.conf), Loader=yaml.FullLoader))
    config.update(dict_args)
    config.var_len = config.data_diff + 1

    exp_id = config.get('exp_id', None)
    if exp_id is None:
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

    train(config, dataset)

if __name__ == '__main__':
    main()











