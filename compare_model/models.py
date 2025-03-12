import torch
from torch import nn
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, LinearRegression
from fbprophet import Prophet
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.svm import SVR
#from transformer.ts_transformer import Transformer
from compare_model.abstract_model import Machine_Learn, Deep_Learn

"""ML"""
RF_para = {'n_estimators': 20}
XGB_para = {'n_estimators': 20}
Stacking_para = {'n_estimators': 20, 'random_state': 902}
TGRNN_para = {}
"""DL"""
ANN_para = {}
LSTM_para = {'hidden_dim': 16, 'layer': 1, 'bidirectional': False}
Seq2Seq_para = {'hid_dim': 16, 'layer': 2, 'dropout': 0.1}
TF_para = {'num_head': 4, 'd_model': 16, 'num_layers': 4, 'dropout': 0.1}
DLinear_para = {'kernel_size': 7, 'individual': False}
"""1"""


class MA():
    def __init__(self, config):
        self.input_len = config.input_len
        self.pred_len = config.pred_len

    def fit(self, train_data):
        pass

    def predict(self, test_data):
        all_pred = []
        for node in range(test_data.shape[1]):
            data = test_data[:, node]
            np_pred = self.one_moving_average(data, self.input_len, self.pred_len)
            all_pred.append(np_pred.reshape([-1, 1]))
        all_pred = np.concatenate(all_pred, axis=1)
        return all_pred

    def one_moving_average(self, data, input_len, pred_len):  # 一次移动平均
        # 第一个参数是表格数据，第二个参数是 N 跨度的取值
        assert input_len <= len(data), "N should lower than data's length"
        all_pred = []  # 定义 M 来记录预测计算结果
        for i in range(0, len(data), pred_len):
            if (i + input_len + pred_len) > len(data):  # 控制预测值不超出标签范围，便于误差计算
                break
            input = data[i:i + input_len]
            for j in range(0, pred_len):
                pred = np.array([np.mean(input)])
                input = np.concatenate([input[len(pred):], pred])  # 多步预测
                all_pred.append(pred)
        np_all_pred = np.concatenate(all_pred)
        return np_all_pred

    def two_moving_average(self, data, input_len, pred_len, T=1):  # 二次移动平均
        M1 = self.one_moving_average(data, input_len, pred_len)
        M2 = self.one_moving_average(M1, input_len, pred_len)
        a = 2 * M1[len(M1) - 1] - M2[len(M2) - 1]
        b = (2 / (input_len - 1)) * (M1[len(M1) - 1] - M2[len(M2) - 1])  # 计算b
        X = a + b * T  # 计算 X （预测值）
        return X


class V_AR():
    def __init__(self, config):
        self.input_len = config.input_len
        self.pred_len = config.pred_len

    def fit(self, train_data, d=1):
        self.d = d
        train_data = np.diff(train_data, n=d, axis=0)
        adf_result = []
        for i in range(train_data.shape[1]):
            p = self.adf_test(series=train_data[:, i])
            adf_result.append(p)
        assert sum(adf_result) == 0, "Data has an unit root and is non-stationary"
        self.model = VAR(train_data)
        # self.results = self.model.fit(maxlags=self.input_len, ic='aic')
        self.results = self.model.fit(maxlags=self.input_len-1)
        # print(results.summary())

    # def predict(self, test_data):
    #     lag_order = self.results.k_ar
    #     all_pred = []
    #     for i in range(0, test_data.shape[0], self.pred_len):
    #         data = test_data[i: i + self.input_len]
    #         data_diff = np.diff(data, n=self.d, axis=0)
    #         forecast = self.results.forecast(data_diff[-lag_order:], steps=self.pred_len)
    #         before_pred = data[-1, :].reshape([1, -1])
    #         for j in range(forecast.shape[0]):
    #             if j == 0:
    #                 forecast[j, :] = forecast[j, :] + before_pred
    #             else:
    #                 forecast[j, :] = forecast[j, :] + forecast[j-1, :]
    #         all_pred.append(forecast)
    #     all_pred = np.concatenate(all_pred, axis=0)
    #     return all_pred

    def predict(self, test_data):
        node = test_data.shape[1]
        lag_order = self.results.k_ar
        all_pred = []
        for i in range(0, test_data.shape[0], self.pred_len):
            if (i + self.input_len + self.pred_len) > test_data.shape[0]:  # 控制预测值不超出标签范围，便于误差计算
                break
            data = test_data[i: i + self.input_len]
            data_diff = np.diff(data, n=self.d, axis=0)
            before_pred = data[-1, :].reshape([1, -1])
            forecast = np.zeros([self.pred_len, node])
            for t in range(self.pred_len):
                forecast[t, :] = self.results.forecast(data_diff[-lag_order:], steps=1)
                data_diff = np.concatenate([data_diff[1:], forecast[t, :].reshape((1, -1))], axis=0)
                if t == 0:
                    forecast[t, :] = forecast[t, :] + before_pred
                else:
                    forecast[t, :] = forecast[t, :] + forecast[t - 1, :]
            all_pred.append(forecast)
        all_pred = np.concatenate(all_pred, axis=0)
        return all_pred

    def adf_test(self, series, title=''):
        print(f'Augmented Dickey-Fuller Test: {title}')
        result = adfuller(series, autolag='AIC')
        labels = ['ADF test statistic', 'p-value', '# lags used', '# observations']
        out = pd.Series(result[0:4], index=labels)
        for key, val in result[4].items():
            out[f'critical value ({key})'] = val
        print(out.to_string())
        if result[1] <= 0.05:
            print("Data has no unit root and is stationary")
            return 0
        else:
            print("Data has a unit root and is non-stationary")
            return 1


class ARI_MA():
    def __init__(self, config):
        self.input_len = config.input_len
        self.pred_len = config.pred_len
        self.train_time = 0
        self.pred_time = 0

    def fit(self, train_data):
        pass

    # def predict(self, test_data, p=6, d=1, q=8):
    #     all_pred = []
    #     for n in tqdm(range(test_data.shape[1]), total=test_data.shape[1]):
    #         forecast = []
    #         for i in range(0, test_data.shape[0], self.pred_len):
    #             data = test_data[i: i + self.input_len, n]
    #             model = ARIMA(data, order=(p, d, q))
    #             result = model.fit()
    #             pred = np.zeros((self.pred_len, 1))
    #             for t in range(self.pred_len):
    #                 pred[t, :] = result.forecast(steps=1)
    #                 data = np.hstack([data[1:], pred[t, :]])
    #             forecast.append(pred)
    #         all_pred.append(np.concatenate(forecast, axis=0))
    #
    #     all_pred = np.concatenate(all_pred, axis=1)
    #     return all_pred

    def predict(self, test_data, p=10, d=1, q=10):
        all_pred = []
        for n in tqdm(range(test_data.shape[1]), total=test_data.shape[1]):
            forecast = []
            for i in range(0, test_data.shape[0], self.pred_len):
                if (i + self.input_len + self.pred_len) > test_data.shape[0]:
                    break
                data = test_data[i: i + self.input_len, n]
                model = ARIMA(data, order=(p, d, q))
                train_start_time = time.time()
                result = model.fit()
                train_end_time = time.time()
                self.train_time += (train_end_time - train_start_time)
                pred_s_time = time.time()
                pred = result.forecast(steps=self.pred_len)
                pred_e_time = time.time()
                self.pred_time += pred_e_time - pred_s_time
                forecast.append(pred.reshape([-1, 1]))
            all_pred.append(np.concatenate(forecast, axis=0))

        all_pred = np.concatenate(all_pred, axis=1)
        return all_pred


class prophet():
    def __init__(self, config):
        self.input_len = config.input_len
        self.pred_len = config.pred_len
        self.num_node = config.capacity
        self.train_time = 0
        self.pred_time = 0

    def fit(self, train_data):
        pass

    def predict(self, test_data, date_list):
        # all_pred = []
        # for node in tqdm(range(self.num_node), total=self.num_node):
        #     time = self.time_window(date_list, self.input_len, self.pred_len)
        #     input = self.time_window(test_data[:, node], self.input_len, self.pred_len)
        #     one_node_pred = []
        #     for ds, y in zip(time, input):
        #         df_ds = pd.Series(ds)
        #         df_y = pd.Series(y)
        #         df = pd.concat([df_ds, df_y], axis=1)
        #         df.columns = ['ds', 'y']
        #         pred = np.zeros((self.pred_len, 1))
        #         for t in range(self.pred_len):
        #             model = Prophet()
        #             model.fit(df)
        #             future = model.make_future_dataframe(periods=1, freq='min')
        #             forecast = model.predict(future)[['ds', 'yhat']].iloc[-1]
        #             df = df.drop(df.index[0])
        #             df.loc[df.index[-1] + 1] = [forecast['ds'], forecast['yhat']]
        #             pred[t, :] = forecast['yhat']
        #         one_node_pred.append(pred)
        #     all_pred.append(np.concatenate(one_node_pred, axis=0))
        # return np.concatenate(all_pred, axis=1)
        all_pred = []
        for node in tqdm(range(self.num_node), total=self.num_node):
            data_time = self.time_window(date_list, self.input_len, self.pred_len)
            input = self.time_window(test_data[:, node], self.input_len, self.pred_len)
            one_node_pred = []
            for ds, y in zip(data_time, input):
                df_ds = pd.Series(ds)
                df_y = pd.Series(y)
                df = pd.concat([df_ds, df_y], axis=1)
                df.columns = ['ds', 'y']
                model = Prophet()
                train_start_time = time.time()
                model.fit(df)
                train_end_time = time.time()
                self.train_time += (train_end_time - train_start_time)

                pred_s_time = time.time()
                future = model.make_future_dataframe(periods=self.pred_len, freq='min')
                forecast = model.predict(future)[['ds', 'yhat']].iloc[-self.pred_len:]
                pred_e_time = time.time()
                self.pred_time += (pred_e_time - pred_s_time)

                one_node_pred.append(np.array(forecast['yhat']).reshape((-1, 1)))
            all_pred.append(np.concatenate(one_node_pred, axis=0))
        return np.concatenate(all_pred, axis=1)

    def time_window(self, data, window_size, step):
        data_list = []
        for i in range(0, len(data), step):
            if (i + window_size + step) > len(data):
                break
            data_list.append(data[i:i + window_size])
        return data_list


"""2"""


class RF(Machine_Learn):
    def __init__(self, config, random_state=42):
        super().__init__(config)
        self.n = RF_para['n_estimators']
        self.random_state = random_state  # 将 random_state 存储为实例变量

    def fit(self, x, y):
        """
        :param x: [bs, in_len, 24]
        :param y: [bs, 1, 24]
        """
        # 使用实例变量 self.random_state
        model = RandomForestRegressor(n_estimators=self.n, random_state=self.random_state)
        super().fit_model(x, y, model)

    def predict(self, test_x):
        all_node_pred = super().predict(test_x)
        return all_node_pred




class XGB(Machine_Learn):
    def __init__(self, config, seed=42,n_estimators=100,):
        super().__init__(config)
        # self.n = XGB_para['n_estimators']
        self.n = n_estimators
        self.seed = seed  # 将 seed 作为实例变量

    def fit(self, x, y):
        """
        :param x: [bs, in_len, 24]
        :param y: [bs, 1, 24]
        """
        # 使用 seed 传递给 XGBRegressor 的 random_state
        model = XGBRegressor(n_estimators=self.n, random_state=self.seed)
        super().fit_model(x, y, model)

    def predict(self, test_x):
        all_node_pred = super().predict(test_x)
        return all_node_pred



class SVM(Machine_Learn):
    def __init__(self, config,seed,n_estimators,c):
        super().__init__(config)
        self.seed = seed
        self.n_estimators = n_estimators
        self.c = c
    def fit(self, x, y):
        """
        :param x:[bs, in_len, 24]
        :param y:[bs, 1, 24]
        """
        model = SVR(kernel='linear', tol=1e-4, C=self.c, epsilon=0.04 )
        # model = SVR(kernel='linear', tol=1e-4, C=0.02, epsilon=0.08)
        super().fit_model(x, y, model)

    def predict(self, test_x):
        all_node_pred = super().predict(test_x)
        return all_node_pred


class Stacking(Machine_Learn):
    def __init__(self, config):
        super().__init__(config)
        clf1 = LinearRegression()
        clf2 = RandomForestRegressor()
        clf3 = KNeighborsRegressor()
        clf4 = SVR()
        clf5 = XGBRegressor()
        clf6 = MLPRegressor()
        estimators = [("LR", clf1), ("RF", clf2), ("KNN", clf3), ("SVR", clf4), ("XGB", clf5), ("MLP", clf6)]
        final_estimator = RandomForestRegressor(n_estimators=Stacking_para['n_estimators'],
                                                random_state=Stacking_para['random_state'])
        self.clf_ST = StackingRegressor(estimators=estimators, final_estimator=final_estimator)

    def fit(self, x, y):
        """
        :param x:[bs, in_len, 24]
        :param y:[bs, 1, 24]
        """
        model = self.clf_ST
        super().fit_model(x, y, model)

    def predict(self, test_x):
        all_node_pred = super().predict(test_x)
        return all_node_pred


class TGRNN(Machine_Learn):
    def __init__(self, config):
        super().__init__(config)

    def fit(self, x, y):
        """
        :param x:[bs, in_len, 24]
        :param y:[bs, 1, 24]
        """
        model = Ridge(alpha=20, tol=0.1)
        super().fit_model(x, y, model)

    def predict(self, test_x):
        all_node_pred = super().predict(test_x)
        return all_node_pred


"""3"""


class ANN(Deep_Learn):
    def __init__(self, config):
        super().__init__(config, config.epochs, config.lr, config.patient, config.device, config.out_path)
        self.input_len = config.input_len
        self.pred_len = config.pred_len
        self.num_node = config.capacity

    def fit(self, train_dataloader, model_class=None):
        model_class = ANN_model
        super().fit(train_dataloader, model_class)

    def predict(self, test_dataloader, model_class=None):
        model_class = ANN_model
        all_pred, all_gt = super().predict(test_dataloader, model_class)
        return all_pred, all_gt


class LSTM(Deep_Learn):
    def __init__(self, config):
        super().__init__(config, config.epochs, config.lr, config.patient, config.device, config.out_path)
        self.input_len = config.input_len
        self.pred_len = config.pred_len
        self.num_node = config.capacity

    def fit(self, train_dataloader, model_class=None):
        model_class = LSTM_model
        super().fit(train_dataloader, model_class)

    def predict(self, test_dataloader, model_class=None):
        model_class = LSTM_model
        all_pred, all_gt = super().predict(test_dataloader, model_class)
        return all_pred, all_gt


class Seq2Seq(Deep_Learn):
    def __init__(self, config):
        super().__init__(config, config.epochs, config.lr, config.patient, config.device, config.out_path)
        self.input_len = config.input_len
        self.pred_len = config.pred_len
        self.num_node = config.capacity

    def fit(self, train_dataloader, model_class=None):
        model_class = Seq2seq_model
        super().fit(train_dataloader, model_class)

    def predict(self, test_dataloader, model_class=None):
        model_class = Seq2seq_model
        all_pred, all_gt = super().predict(test_dataloader, model_class)
        return all_pred, all_gt


class TF(Deep_Learn):
    def __init__(self, config):
        super().__init__(config, config.epochs, config.lr, config.patient, config.device, config.out_path)
        self.input_len = config.input_len
        self.pred_len = config.pred_len
        self.num_node = config.capacity

    def fit(self, train_dataloader, model_class=None):
        model_class = TF_model
        super().fit(train_dataloader, model_class)

    def predict(self, test_dataloader, model_class=None):
        model_class = TF_model
        all_pred, all_gt = super().predict(test_dataloader, model_class)
        return all_pred, all_gt


class DLinear(Deep_Learn):
    def __init__(self, config):
        super().__init__(config, config.epochs, config.lr, config.patient, config.device, config.out_path)
        self.input_len = config.input_len
        self.pred_len = config.pred_len
        self.num_node = config.capacity

    def fit(self, train_dataloader, model_class=None):
        model_class = DLinear_model
        super().fit(train_dataloader, model_class)

    def predict(self, test_dataloader, model_class=None):
        model_class = DLinear_model
        all_pred, all_gt = super().predict(test_dataloader, model_class)
        return all_pred, all_gt


"""DL models"""


class ANN_model(nn.Module):
    def __init__(self, config):
        super().__init__()
        # self.fc_node = nn.Linear(in_features=in_dim, out_features=out_dim)
        input_len = config.input_len
        pred_len = config.pred_len
        in_dim = config.var_len
        out_dim = config.out_dim
        self.fc1 = nn.Linear(in_features=input_len, out_features=input_len * 2)
        self.fc2 = nn.Linear(in_features=input_len * 2, out_features=input_len)
        self.fc3 = nn.Linear(in_features=input_len, out_features=pred_len)
        self.relu = nn.ReLU()

    def forward(self, x):
        "x: [bs, seq_len]"
        # x = self.fc_node(x)
        # x = torch.permute(x, (0, 2, 1))
        # out = self.relu(self.fc1(x))
        # out = self.relu(self.fc2(out))
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        out = torch.squeeze(out)
        return out


class LSTM_model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_len = config.input_len
        self.pred_len = config.pred_len
        self.diff = config.data_diff
        in_dim = config.var_len
        out_dim = config.out_dim
        hidden_dim = LSTM_para['hidden_dim']
        layer = LSTM_para['layer']
        bidirectional = LSTM_para['bidirectional']

        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=hidden_dim,
                            num_layers=layer, batch_first=True, bidirectional=bidirectional)
        if bidirectional:
            self.fc = nn.Linear(in_features=hidden_dim * 2, out_features=hidden_dim // 2)
        else:
            self.fc = nn.Linear(in_features=hidden_dim, out_features=hidden_dim // 2)
        self.out_fc = nn.Linear(in_features=hidden_dim // 2, out_features=out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        "x: [bs, seq_len]"
        x = torch.unsqueeze(x, dim=-1)   # [bs, seq_len, 1]
        bz = x.shape[0]
        if self.diff != 0:
            diff_data = []
            inputs_diff = x
            for d in range(self.diff):
                inputs_diff = inputs_diff[:, 1:, :] - inputs_diff[:, :-1, :]
                inputs_diff = torch.cat((torch.zeros(bz, 1, 1).to(x.device), inputs_diff), 1)
                diff_data.append(inputs_diff)
            inputs_diff = torch.cat(diff_data, dim=2)
            x = torch.cat((x, inputs_diff), 2)

        feat, _ = self.lstm(x)
        # feat = self.relu(self.fc(feat))
        feat = self.fc(feat)
        feat = self.out_fc(feat)
        return feat[:, -self.pred_len:, 0]


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 1
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=False)

    def forward(self, input_seq):
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        h_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(input_seq.device)
        c_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(input_seq.device)
        output, (h, c) = self.lstm(input_seq, (h_0, c_0))
        return h, c


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 1
        self.lstm = nn.LSTM(input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=False)
        # self.linear = nn.Linear(self.hidden_size, self.input_size)

    def forward(self, input_seq, h, c):
        # input_seq(batch_size, input_size)
        input_seq = input_seq.unsqueeze(1)
        output, (h, c) = self.lstm(input_seq, (h, c))
        # output(batch_size, seq_len, num * hidden_size)
        # pred = self.linear(output.squeeze(1))  # pred(batch_size, 1, output_size)
        return output.squeeze(1), h, c


class Seq2seq_model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_len = config.input_len
        self.output_size = config.pred_len
        self.in_dim = config.var_len
        self.out_dim = config.out_dim
        self.diff = config.data_diff
        hid_dim = Seq2Seq_para['hid_dim']
        layers = Seq2Seq_para['layer']

        self.Encoder = Encoder(self.in_dim, hid_dim, layers)
        self.Decoder = Decoder(hid_dim, hid_dim, layers, self.output_size)
        self.embedding = nn.Linear(self.in_dim, hid_dim)
        self.out_fc = nn.Linear(hid_dim, self.out_dim)
        self.dropout = nn.Dropout(Seq2Seq_para['dropout'])

    def forward(self, input_seq):
        "x: [bs, seq_len]"
        input_seq = torch.unsqueeze(input_seq, dim=-1)  # [bs, seq_len, 1]
        bz = input_seq.shape[0]
        if self.diff != 0:
            diff_data = []
            inputs_diff = input_seq
            for d in range(self.diff):
                inputs_diff = inputs_diff[:, 1:, :] - inputs_diff[:, :-1, :]
                inputs_diff = torch.cat((torch.zeros(bz, 1, 1).to(input_seq.device), inputs_diff), 1)
                diff_data.append(inputs_diff)
            inputs_diff = torch.cat(diff_data, dim=2)
            input_seq = torch.cat((input_seq, inputs_diff), 2)

        target_len = self.output_size  # 预测步长
        batch_size, seq_len, _ = input_seq.shape[0], input_seq.shape[1], input_seq.shape[2]
        h, c = self.Encoder(input_seq)
        outputs = torch.zeros(batch_size, self.out_dim, self.output_size).to(input_seq.device)

        input_seq = self.dropout(self.embedding(input_seq))

        decoder_input = input_seq[:, -1, :]
        for t in range(target_len):
            decoder_output, h, c = self.Decoder(decoder_input, h, c)
            outputs[:, :, t] = self.out_fc(decoder_output)
            decoder_input = decoder_output

        return torch.squeeze(outputs)


class TF_model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_len = config.input_len
        self.pred_len = config.pred_len

        n_encoder_inputs = config.var_len
        self.diff = config.data_diff
        out_dim = config.out_dim
        n_head = TF_para['num_head']
        d_model = TF_para['d_model']
        n_layers = TF_para['num_layers']
        dropout = TF_para['dropout']

        self.input_pos_embedding = torch.nn.Embedding(256, embedding_dim=d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dropout=dropout,
            dim_feedforward=4 * d_model,

        )

        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.input_projection = nn.Linear(n_encoder_inputs, d_model)
        # self.decode = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Linear(d_model // 2, out_dim))
        self.decode = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.Linear(d_model // 2, out_dim))

    def encode(self, src):
        # src: [bs, seq_len]
        src = torch.unsqueeze(src, dim=-1)  # [bs, in_len, dim]

        bz = src.shape[0]
        if self.diff != 0:
            diff_data = []
            inputs_diff = src
            for d in range(self.diff):
                inputs_diff = inputs_diff[:, 1:, :] - inputs_diff[:, :-1, :]
                inputs_diff = torch.cat((torch.zeros(bz, 1, 1).to(src.device), inputs_diff), 1)
                diff_data.append(inputs_diff)
            inputs_diff = torch.cat(diff_data, dim=2)
            src = torch.cat((src, inputs_diff), 2)

        src_start = self.input_projection(src).permute(1, 0, 2)  # [in_len, bs, dim]

        in_sequence_len, batch_size = src_start.size(0), src_start.size(1)
        pos_encoder = (
            torch.arange(0, in_sequence_len, device=src.device)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )

        pos_encoder = self.input_pos_embedding(pos_encoder).permute(1, 0, 2)

        src = src_start + pos_encoder

        src = self.encoder(src) + src_start  # [seq_len, bs, 32]

        return src.permute(1, 0, 2)  # [bs, seq_len, 32]

    def forward(self, src):
        out = self.encode(src)  # 只用编码器部分
        out = self.decode(out)  # [bs, seq_len, 1]
        return out[:, -self.pred_len:, 0]


"""DLinear"""


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # x: [bs, t, dim]
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class DLinear_model(nn.Module):
    """
    DLinear
    """

    def __init__(self, config, ks=None, individual=None):
        super(DLinear_model, self).__init__()
        self.Lag = config.input_len
        self.Horizon = config.pred_len
        self.individual = DLinear_para['individual'] if (individual is None) else individual

        # Decompsition Kernel Size
        kernel_size = DLinear_para['kernel_size'] if (ks is None) else ks
        self.decompsition = series_decomp(kernel_size)

        self.channels = config.var_len
        self.feature1 = nn.Linear(config.var_len, 6)
        self.feature2 = nn.Linear(6, 1)
        if self.individual:
            # individual linear layer for each para
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            self.Linear_Decoder = nn.ModuleList()
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.Lag, self.Horizon))
                self.Linear_Seasonal[i].weight = nn.Parameter((1 / self.Lag) * torch.ones([self.Horizon, self.Lag]))
                self.Linear_Trend.append(nn.Linear(self.Lag, self.Horizon))
                self.Linear_Trend[i].weight = nn.Parameter((1 / self.Lag) * torch.ones([self.Horizon, self.Lag]))
                self.Linear_Decoder.append(nn.Linear(self.Lag, self.Horizon))
        else:
            self.Linear_Seasonal = nn.Linear(self.Lag, self.Horizon)
            self.Linear_Trend = nn.Linear(self.Lag, self.Horizon)
            self.Linear_Decoder = nn.Linear(self.Lag, self.Horizon)
            self.Linear_Seasonal.weight = nn.Parameter((1 / self.Lag) * torch.ones([self.Horizon, self.Lag]))
            self.Linear_Trend.weight = nn.Parameter((1 / self.Lag) * torch.ones([self.Horizon, self.Lag]))

    def forward(self, x):
        # x: [Batch, Input length]

        x = torch.unsqueeze(x, dim=-1)  # [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.Horizon],
                                          dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.Horizon],
                                       dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        x = self.feature1(x.permute(0, 2, 1))
        x = self.feature2(x)  # to [Batch, Output length, Channel]
        return x[..., -1]
