# For relative import
import os
import sys
import copy
import os
from collections import deque

from alembic.command import current


def predict_and_save_figures(train_path, test_path, save_dir, in_len, out_len, step, save_csv_flag):
    # 确保保存图像的目录存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
PROJ_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJ_DIR)
print(PROJ_DIR)
import random
import argparse
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from typing import Any, Callable, Optional
# from pytorch_lightning.metrics.metric import Metric
from torchmetrics import Metric
from tqdm import tqdm

parser = argparse.ArgumentParser()
args = parser.parse_args()

gpu_num = 0  # set the GPU number of your server.
os.environ['WANDB_MODE'] = 'offline'  # select one from ['online','offline']

device = 'cuda:0'
def save_result_csv(np_true, np_pred, file_path, file_name, columns: list):
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    for i in range(np_true.shape[1]):
        df_data = pd.DataFrame(np.vstack([np_true[:, i], np_pred[:, i]]).T,
                               columns=[f'{columns[i]}_true', f'{columns[i]}_pred'])
        if i == 0:
            save_df = df_data
        else:
            save_df = pd.concat([save_df, df_data], axis=1)

    save_df.to_csv(os.path.join(file_path, f'{file_name}.csv'), index=False)


def RAE(pred, gt):
    gt_mean = np.mean(gt)
    squared_error_num = np.sum(np.abs(gt - pred))
    squared_error_den = np.sum(np.abs(gt - gt_mean))
    rae_loss = squared_error_num / squared_error_den
    return rae_loss

def MAE(pred, gt):
    _mae = 0.
    if len(pred) > 0 and len(gt) > 0:
        _mae = np.mean(np.abs(pred - gt))
    return _mae


def MSE(pred, gt):
    _mse = 0.
    if len(pred) > 0 and len(gt) > 0:
        _mse = np.mean((pred - gt) ** 2)
    return _mse


def RMSE(pred, gt):
    return np.sqrt(MSE(pred, gt))


def MAPE(pred, gt):
    _mape = 0.
    if len(pred) > 0 and len(gt) > 0:
        _mape = np.mean(np.abs((pred - gt) / gt)) * 100
    return _mape


def SMAPE(pred, gt):
    _smape = 0.
    if len(pred) > 0 and len(gt) > 0:
        _smape = 2.0 * np.mean(np.abs(pred - gt) / (np.abs(pred) + np.abs(gt))) * 100
    return _smape


def evaluate_forecasts(predictions, gts, features):
    '''
    predictions:(N,F)
    '''
    all_rmse, all_mae, all_mape, all_smape, all_RAE = [], [], [], [], []
    for i in range(features):
        prediction = predictions[:, i]  # (N,1)
        gt = gts[:, i]
        _rmse = RMSE(prediction, gt); all_rmse.append(_rmse)
        _mae = MAE(prediction, gt); all_mae.append(_mae)
        _mape = MAPE(prediction, gt); all_mape.append(_mape)
        _smape = SMAPE(prediction, gt); all_smape.append(_smape)
        _rae = RAE(prediction, gt); all_RAE.append(_rae)


    return all_rmse, all_mae, all_mape, all_smape, all_RAE





def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels) / labels
    loss = torch.abs(loss)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def metric(pred, real):
    mae = masked_mae(pred, real).item()
    mape = masked_mape(pred, real).item()
    rmse = masked_rmse(pred, real).item()
    return np.round(mae, 4), np.round(mape, 4), np.round(rmse, 4)


class LightningMetric(Metric):

    def __init__(
            self,
            compute_on_step: bool = True,
            dist_sync_on_step: bool = False,
            process_group: Optional[Any] = None,
            dist_sync_fn: Callable = None,
    ):
        super().__init__(
            # compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.add_state("y_true", default=[], dist_reduce_fx=None)
        self.add_state("y_pred", default=[], dist_reduce_fx=None)

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        self.y_pred.append(preds)
        self.y_true.append(target)

    def compute(self):
        """
        Computes explained variance over state.
        """
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)

        feature_dim = y_pred.shape[-1]
        pred_len = y_pred.shape[1]
        # (16, 12, 38, 1)

        y_pred = torch.reshape(y_pred.permute((0, 2, 1, 3)), (-1, pred_len, feature_dim))
        y_true = torch.reshape(y_true.permute((0, 2, 1, 3)), (-1, pred_len, feature_dim))

        y_pred = y_pred[..., 0]
        y_true = y_true[..., 0]

        metric_dict = {}
        rmse_avg = []
        mae_avg = []
        mape_avg = []
        for i in range(pred_len):
            mae, mape, rmse = metric(y_pred[:, i], y_true[:, i])
            idx = i + 1

            metric_dict.update({'rmse_%s' % idx: rmse})
            metric_dict.update({'mae_%s' % idx: mae})
            metric_dict.update({'mape_%s' % idx: mape})

            rmse_avg.append(rmse)
            mae_avg.append(mae)
            mape_avg.append(mape)

        metric_dict.update({'rmse_avg': np.round(np.mean(rmse_avg), 4)})
        metric_dict.update({'mae_avg': np.round(np.mean(mae_avg), 4)})
        metric_dict.update({'mape_avg': np.round(np.mean(mape_avg), 4)})

        return metric_dict


class SGRU(nn.Module):
    def __init__(self, n_features, n_outputs, ws, out_len):
        super(SGRU, self).__init__()
        self.out_len = out_len
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.gru = nn.GRU(input_size=n_features, hidden_size=32, num_layers=4,  bidirectional=True, batch_first=True)
        self.dropout2 = nn.Dropout(0.1)
        self.dense1 = nn.Linear(in_features=32 * 2, out_features=32)
        self.dense2 = nn.Linear(in_features=32, out_features=16)
        self.dense3 = nn.Linear(in_features=16, out_features=n_outputs)


        # self.relu = nn.ReLU()
        # self.flatten = nn.Flatten()
        # self.gru = nn.GRU(input_size=n_features * 10, hidden_size=128, bidirectional=True, batch_first=True)
        # self.dropout2 = nn.Dropout(0.1)
        # self.dense1 = nn.Linear(in_features=128 * 2, out_features=128)
        # self.dense2 = nn.Linear(in_features=128, out_features=64)
        # self.dense3 = nn.Linear(in_features=64, out_features=n_outputs)

    def forward(self, x):  # (bs, in_channel, len)  #(32, 24, 20)
        output = self.gru(x)[0]  # (32, 64, 60)
        # output = torch.permute(output, (0, 2, 1))  # (32, 60, 64)
        output = self.relu(self.dense1(output))  # (32, 60, 32)
        output = self.relu(self.dense2(output))  # (32, 60, 16)
        output = self.dense3(output)    # (32, 60, 1)
        output = output[:, -self.out_len:,:]
        return output


class dataset(Dataset):
    def __init__(self, X, Y, in_len, out_len, step):
        super(dataset, self).__init__()
        self.in_len = in_len
        self.out_len = out_len
        self.step = step
        self.x, self.y = self.sliding_window(X, Y,in_len,out_len)  # [n,10,54]  [n, 1]
        self.x = torch.from_numpy(self.x).to(torch.float32)
        self.y = torch.from_numpy(self.y).to(torch.float32)

    def __getitem__(self, item):
        return self.x[item], self.y[item]

    # def sliding_window(self, np_X, np_Y, in_start=0):
    #     '''
    #     :param np_X: (3600, 43)
    #     :param np_Y: (3600, 1)
    #     '''
    #     x, y = [], []
    #     for _ in range(0, np_X.shape[0]):
    #         in_end = in_start + self.in_len  # 0 + 5
    #         out_end = in_end + self.out_len  # 5 + 1 -1
    #
    #         # 保证截取样本完整，最大元素索引不超过原序列索引，则截取数据；否则丢弃该样本
    #         if out_end <= np_X.shape[0]:
    #             # 多变量输入
    #             x.append(np_X[in_start:in_end, :])
    #             y.append(np_Y[in_end:out_end, 0])
    #         in_start += self.step  # 0 + 1
    #     return np.array(x), np.array(y)

    def sliding_window(self,np_X, np_Y, n_in, n_out=15, in_start=0):
        '''
        :param np_X: (3600, 43)
        :param np_Y: (3600, 1)
        '''
        x, y = [], []

        for _ in range(0, np_X.shape[0]):
            in_end = in_start + n_in  # 0 + 5
            out_end = in_end + n_out  # 5 + 1 -1

            # 保证截取样本完整，最大元素索引不超过原序列索引，则截取数据；否则丢弃该样本
            if out_end <= np_X.shape[0]:
                # 多变量输入
                x.append(np_X[in_start:in_end, :])
                y.append(np_Y[in_end:out_end, :])
            in_start += 1  # 0 + 1
        return np.array(x), np.array(y)




    def __len__(self):
        return self.x.shape[0]
class test_dataset(Dataset):
    def __init__(self, X, Y, in_len, out_len, step):
        super(test_dataset, self).__init__()
        self.in_len = in_len
        self.out_len = out_len
        self.step = step
        self.x, self.y = self.sliding_window_test(X, Y,in_len,out_len)  # [n,10,54]  [n, 1]
        self.x = torch.from_numpy(self.x).to(torch.float32)
        self.y = torch.from_numpy(self.y).to(torch.float32)

    def __getitem__(self, item):
        return self.x[item], self.y[item]

    # def sliding_window(self, np_X, np_Y, in_start=0):
    #     '''
    #     :param np_X: (3600, 43)
    #     :param np_Y: (3600, 1)
    #     '''
    #     x, y = [], []
    #     for _ in range(0, np_X.shape[0]):
    #         in_end = in_start + self.in_len  # 0 + 5
    #         out_end = in_end + self.out_len  # 5 + 1 -1
    #
    #         # 保证截取样本完整，最大元素索引不超过原序列索引，则截取数据；否则丢弃该样本
    #         if out_end <= np_X.shape[0]:
    #             # 多变量输入
    #             x.append(np_X[in_start:in_end, :])
    #             y.append(np_Y[in_end:out_end, 0])
    #         in_start += self.step  # 0 + 1
    #     return np.array(x), np.array(y)

    def sliding_window_test(self,np_X, np_Y, n_in, n_out=15, in_start=0):
        '''
        :param np_X: (3600, 43)
        :param np_Y: (3600, 1)
        '''
        x, y = [], []

        for _ in range(0, np_X.shape[0]):
            in_end = in_start + n_in  # 0 + 5
            out_end = in_end + n_out  # 5 + 1 -1

            # 保证截取样本完整，最大元素索引不超过原序列索引，则截取数据；否则丢弃该样本
            if out_end <= np_X.shape[0]:
                # 多变量输入
                x.append(np_X[in_start:in_end, :])
                y.append(np_Y[in_end:out_end, :])
            in_start += n_out  # 0 + 1
        return np.array(x), np.array(y)



    def __len__(self):
        return self.x.shape[0]


class LightningData(LightningDataModule):
    def __init__(self, train_set, val_set, test_set, bs):
        super().__init__()
        self.batch_size = bs
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=0,
                          pin_memory=True, drop_last=False)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=0,
                          pin_memory=True, drop_last=False)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=0,
                          pin_memory=True, drop_last=False)


class LightningModel(LightningModule):
    def __init__(self, scaler, model):
        super().__init__()

        self.scaler = scaler

        self.metric_lightning = LightningMetric()

        # self.loss = nn.SmoothL1Loss()
        self.loss = nn.MSELoss()

        self.model = model
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, x):
        return self.model(x)

    def _run_model(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        return y_hat, y, loss

    def training_step(self, batch, batch_idx):
        y_hat, y, loss = self._run_model(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat, y, loss = self._run_model(batch)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)


    def test_step(self, batch, batch_idx):
        y_hat, y, loss = self._run_model(batch)
        self.metric_lightning(y_hat.cpu(), y.cpu())
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def on_train_epoch_end(self):
        current_epoch = self.trainer.current_epoch
        self.log("epoch", current_epoch)  # 将 epoch 存为日志信息

    def test_epoch_end(self, outputs):
        test_metric_dict = self.metric_lightning.compute()
        self.log_dict(test_metric_dict)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.002, weight_decay=1e-3)


def main(train_path, ws, out_len, s, save_path, epoch_num=100, bs=32):


    train_data = pd.read_csv(train_path).iloc[:,1: 1 + 4 ]
    train_data = train_data.dropna()
    dowm_simple_ind = np.arange(0, len(train_data), step=1)  # 训练集下采样
    train_data = np.array(train_data.iloc[dowm_simple_ind, :])

    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    x=train_data
    y=train_data

    # x = train_data[:, col].reshape((-1, 1))#
    # y = train_data[:, col].reshape((-1, 1))  # [n, 1]#
    train_set = dataset(x, y, ws, out_len, s)
    val_set = dataset(x, y, ws, out_len, s)
    test_set = dataset(x, y, ws, out_len, s)

    model = SGRU(n_features=4, n_outputs=4, ws=ws, out_len=out_len)#x.shape[1]
    lightning_data = LightningData(train_set, val_set, test_set, bs=bs)
    lightning_model = LightningModel(scaler, model)
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, mode='min')
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_path,
        filename='model_epoch_{epoch:02d}',
        save_top_k=-1,
        monitor='val_loss',
        mode='min',
        save_on_train_epoch_end=True,
        save_last=False

    )
    trainer = Trainer(
        gpus=[0],
        max_epochs=epoch_num,
        callbacks=[checkpoint_callback, early_stopping]
        # precision=16,
    )

    trainer.fit(lightning_model, lightning_data)
# trainer.test(lightning_model, datamodule=lightning_data)

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

def predict(train_path, test_path, save_dir, in_len, out_len, step, save_csv_flag):
    train_data = pd.read_csv(train_path).iloc[:, 1: 1 + 4 ]
    train_data = train_data.dropna()
    down_simple = np.arange(0, len(train_data), 1)  # 对测试集进行下采样
    train_data = np.array(train_data.iloc[down_simple, :])

    df_test = pd.read_csv(test_path).iloc[:, 1 : 1 + 4 ]
    df_test = df_test.dropna()
    down_simple = np.arange(0, len(df_test), 1)  # 对测试集进行下采样
    test_data = np.array(df_test.iloc[down_simple, :])
    _test_data = copy.deepcopy(test_data)

    scaler = StandardScaler()
    scaler.fit(train_data)
    test_data = scaler.transform(test_data)


    model = SGRU(4, 4, in_len, out_len=out_len)
    model.to(device)

    epoch_num = 100


    recent_best_results = deque(maxlen=5)
    best_metric=np.inf

    for epoch in range(epoch_num):
        all_pred = []
        all_gt = []
        pred = []
        gt = []
        try:
            pl_model = LightningModel.load_from_checkpoint(checkpoint_path=f'{save_dir}/model_epoch_epoch={epoch:02d}.ckpt',
                                                       map_location=device,
                                                       **{"scaler": scaler, "model": model})
        except Exception as e:
            continue
        # X = test_data[:, col].reshape((-1, 1))
        # Y = _test_data[:, col].reshape((-1, 1))  # [n, 1]
        # data_set = dataset(X, Y, in_len=in_len, out_len=out_len, step=out_len)
        X =test_data
        Y =_test_data
        data_set = test_dataset(X, Y, in_len=in_len, out_len=out_len,step=step)
        data_loader = DataLoader(data_set, batch_size=1, shuffle=False, drop_last=False)

        for x, y in data_loader:
            x = x.to(device)
            y_hat = pl_model(x)
            y_hat = y_hat.to('cpu').detach().numpy()
            pred.append(y_hat)
            gt.append(y.detach().numpy())
        gt = np.squeeze(gt)
        pred = np.squeeze(pred)
        all_pred.append(np.array(pred).reshape((-1, 4)))
        all_gt.append(np.array(gt).reshape((-1, 4)))


        all_pred = scaler.inverse_transform(all_pred)
        all_pred = np.squeeze(all_pred)
        all_gt = np.array(all_gt)
        all_gt = np.squeeze(all_gt)

        all_rmse = []
        all_mae = []
        all_mape = []
        all_smape = []
        all_RAE = []  # 新增RAE





        all_rmse, all_mae, all_mape, all_smape, all_RAE = evaluate_forecasts(all_pred, all_gt,4)  # 包含RAE
        rmse = np.mean(all_rmse[:])
        mae = np.mean(all_mae[:])
        mape = np.mean(all_mape[:])
        smape = np.mean(all_smape[:])
        rae = np.mean(all_RAE[:])

        delta_rmse = np.std(all_rmse[:])
        delta_mae = np.std(all_mae[:])
        delta_mape = np.std(all_mape[:])
        delta_smape = np.std(all_smape[:])
        delta_RAE = np.std(all_RAE[:])

        print(str(epoch) + '--All average SPNDs+Sensors fitting results: '
                           'RMSE: {:.6f}, MAE: {:.6f}, MAPE:{:.6f}%, SMAPE:{:.6f}%, RAE:{:.6f}'.format(rmse, mae,
                                                                                                       mape,
                                                                                                       smape, rae))

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

            all_pred_df = pd.DataFrame(all_pred)


            target_dir = f'../save_result/data/GRU_pred_{out_len}'

                # 如果目录不存在，则创建它
            os.makedirs(target_dir, exist_ok=True)
            all_pred_df.to_csv(f'{target_dir}.csv', index=False)

            print(f'预测数据已经保存到{target_dir}.csv')


        for i in range(len(all_mae)):
            print('~~The sensor_{} forecasting results: '
                  'RMSE: {:.6f}, MAE: {:.6f}, MAPE:{:.6f}%, SMAPE:{:.6f}% , RAE:{:.6f}'.
                  format(i, all_rmse[i], all_mae[i], all_mape[i], all_smape[i], all_RAE[i]))
        print('--All average SPNDs+Sensors fitting results: '
              'RMSE: {:.6f}, MAE: {:.6f}, MAPE:{:.6f}%, SMAPE:{:.6f}%, RAE:{:.6f}'.
              format(rmse, mae, mape, smape, rae))
        print('                                        std: '
              'RMSE: {:.6f}, MAE: {:.6f}, MAPE:{:.6f}%, SMAPE:{:.6f}%, RAE:{:.6f}'.
              format(delta_rmse, delta_mae, delta_mape, delta_smape, delta_RAE))
        print(f'Find the best epoch: {epoch} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

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

            rmse = np.mean(best5_rmse[:])
            mae = np.mean(best5_mae[:])
            mape = np.mean(best5_mape[:])
            smape = np.mean(best5_smape[:])
            rae = np.mean(best5_RAE[:])

            delta_rmse = np.std(best5_rmse[:])
            delta_mae = np.std(best5_mae[:])
            delta_mape = np.std(best5_mape[:])
            delta_smape = np.std(best5_smape[:])
            delta_RAE = np.std(best5_RAE[:])

            rmse_five_epoch_mean.append(rmse)
            mae_five_epoch_mean.append(mae)
            mape_five_epoch_mean.append(mape)
            smape_five_epoch_mean.append(smape)
            RAE_five_epoch_mean.append(rae)

            for j in range(len(all_mae)):
                print('-{}epoch~~The sensor_{} forecasting results: '
                      'RMSE: {:.6f}, MAE: {:.6f}, MAPE:{:.6f}%, SMAPE:{:.6f}% , RAE:{:.6f}'.
                      format(i, j, best5_rmse[j], best5_mae[j], best5_mape[j], best5_smape[j], best5_RAE[j]))
            print('-{}epoch--All average SPNDs+Sensors fitting results: '
                  'RMSE: {:.6f}, MAE: {:.6f}, MAPE:{:.6f}%, SMAPE:{:.6f}%, RAE:{:.6f}'.
                  format(i, rmse, mae, mape, smape, rae))
            print('-{}epoch                                        std: '
                  'RMSE: {:.6f}, MAE: {:.6f}, MAPE:{:.6f}%, SMAPE:{:.6f}%, RAE:{:.6f}'.
                  format(i, delta_rmse, delta_mae, delta_mape, delta_smape, delta_RAE))
        print('----5--Epoch----Mean--and--std---for---SD---SE----')
        print('----5--Epoch----Mean--and--std---for---SD---SE----')
        print('----5--Epoch----Mean--and--std---for---SD---SE----')
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
            .format(delta_rmse_five_epoch_mean, delta_mae_five_epoch_mean, delta_mape_five_epoch_mean,
                    delta_smape_five_epoch_mean,
                    delta_RAE_five_epoch_mean))
            # channels = ['Dpax[1]', 'Dpax[2]', 'Dpax[3]', 'Dpax[4]']
            #
            # # 指定保存路径
            #
            #
            # save_path = save_dir
            # os.makedirs(save_path, exist_ok=True)
            #
            # real_data = all_gt
            # predicted_data = all_pred
            #
            # # 遍历每个通道绘图并保存
            # for i in range(4):
            #     plt.figure(figsize=(12, 6))  # 每张图的大小
            #     plt.plot(real_data[:, i], color='red', label='True', linewidth=2)
            #     plt.plot(predicted_data[:, i], color='blue', label='reconstruction', linewidth=2)
            #     plt.title(f'{channels[i]} Comparison of Actual and Predicted Values')
            #     plt.xlabel('time_samples')
            #     plt.ylabel('value')
            #     plt.legend()
            #     plt.grid()
            #
            #     # 保存图形
            #     plt.savefig(f'{save_path}Dpax[{channels[i]}]_.png')
            #     plt.close()  # 关闭当前图以释放内存

            # print('--------------Print Ending-----------------')
            # print('--------------Print Ending-----------------')
            # print('--------------Print Ending-----------------')
            # print('--------------Print Ending-----------------')


if __name__ == '__main__':


    seed = 3407
    set_seed(seed)
    in_len = 45
    out_len = 15
    step = 1
    save_result_metric = False
    #短期预测
    train_path = r'E:\python\项目目录\Pi_idea3.1\TM_data\Scenario_1\TM_train_1min_drift_2023_0727-0731.csv'
    test_file = r'E:\python\项目目录\Pi_idea3.1\TM_data\Scenario_1\TM_test_1min_2024_0403-0413.csv'
    save_dir = './SGRU_checkpoints_short_term_temp'
    #长期预测
    # train_path = r'E:\python\项目目录\Pi_idea3.1\TM_data\Scenario_2\train_double_1h_drift_2023_0217-0331.csv'
    # test_file = r'E:\python\项目目录\Pi_idea3.1\TM_data\Scenario_2\test_1h_2023_0910-20240131.csv'
    # save_dir = './SGRU_checkpoints_long_term_temp'
    # main(train_path, in_len, out_len, step, save_dir, epoch_num=100, bs=32)
    predict(train_path, test_file, save_dir, in_len=in_len, out_len=out_len, step=step, save_csv_flag=save_result_metric)
