import os
from audioop import minmax

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from logging import getLogger
from tqdm import tqdm
import re
# from uhv_dataset import time_dict
from fastdtw import fastdtw
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from PST_utils import one_zero_normalization

# from minepy import MINE

add_test_data2train = False  # 是否加入部分测试集数据进行上采样并参与训练
min_I = 0
max_I = 1
min_P = 0
max_P = 1

class ListDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class I_Dataset():

    def __init__(  # 这里的部分参数是main_kfold中未定义的
            self,
            data_path,
            filename,
            capacity,
            batch_size,
            MinMaxNormalization,
            weight_adj_epsilon,
            K,
            ind,
            Multi_Graph_num,
            binary,
            train_normal_flag,
            test_normal_flag,
            num_workers,
            pad_with_last_sample=False,
            size=None,
            random=False,
            graph_path=None,
            graph_file=None,

    ):

        super().__init__()
        self.data_path = data_path
        self.filename = filename
        self.capacity = capacity
        self.MinMaxNormalization = MinMaxNormalization

        self.batch_size = batch_size
        self.weight_adj_epsilon = weight_adj_epsilon
        self.K = K
        self.ind = ind
        self.Multi_Graph_num = Multi_Graph_num
        self.binary = binary
        self.train_normal_flag = train_normal_flag
        self.test_normal_flag = test_normal_flag
        self.num_workers = num_workers
        self.pad_with_last_sample = pad_with_last_sample
        self.random = random

        self.input_len = size[0]
        self.output_len = size[1]

        self.start_col = 0
        self.graph_path = graph_path
        self.graph_file = graph_file

        self.__read_data__()

    def __read_data__(self):
        df = pd.read_csv(os.path.join(self.data_path, self.filename))  # (n,f)

        if self.filename == 'unit2_2401-2403_I+P_shutdown.csv':  # 最后500个点进行上采样
            np_low_data = np.array(df.iloc[-500:, :])
            new_np_data = self.upsample(np_low_data, rate=5)
            new_df_data = pd.DataFrame(new_np_data, columns=df.columns)
            df = pd.concat([df.iloc[:-500, :], new_df_data], axis=0, ignore_index=True)
            test_path = './data/unit2_2401-2403_I+P_startup3.csv'
            test_data_rate = 0.3
            df_test = pd.read_csv(test_path)
            df_test = df_test.iloc[:int(df_test.shape[0] * test_data_rate), :]
            df = pd.concat([df_test, df], axis=0, ignore_index=True)

        df_raw_x = df.iloc[:, 1: 7 + 1]  # 第一列是time
        # df_raw_x = df.iloc[:, 24 + 1 + 4:24 + 1 + 4 + 24]
        # df_raw_I = df.iloc[:, 1:24 + 1]  # 第一列是time
        # df_raw_P = df.iloc[:, 24 + 1 + 4:24 + 1 + 4 + 24]
        # df_raw_x = pd.concat([df_raw_I, df_raw_P], axis=1)
        df_raw_y = df.iloc[:, 1: 7 + 1]
        df_raw_x = df_raw_x.dropna()
        df_raw_y = df_raw_y.dropna()

        if self.filename == "unit4_2308_shutdown_I+P.csv":
            dowm_simple_ind_x = np.arange(0, len(df_raw_x), step=5)  # x训练集下采样
            df_raw_x = df_raw_x.iloc[dowm_simple_ind_x, :]
            # df_raw = df_raw * 1.5
            dowm_simple_ind_y = np.arange(0, len(df_raw_y), step=5)  # y训练集下采样
            df_raw_y = df_raw_y.iloc[dowm_simple_ind_y, :]
        # else:
        #     dowm_simple_ind = np.arange(0, len(df_raw), step=1)  # 训练集下采样
        #     df_raw = df_raw.iloc[dowm_simple_ind, :]
        #     if self.DA_tgt_flag:
        #         df_raw = df_raw.iloc[0:int(len(df_raw) * self.tgt_ratio), :]  # for DA

        if add_test_data2train:
            df_test_data = pd.DataFrame(self.test_data_upsample())
            df_test_data.columns = df_raw_x.columns
            normal_x, normal_y, all_scaler_x, all_scaler_y = self.generate_input_data([df_raw_x, df_test_data],
                                                                                      normal_flag=self.train_normal_flag,MinMaxNormalization=self.MinMaxNormalization)
        else:
            normal_x, normal_y, all_scaler_x, all_scaler_y = self.generate_input_data(df_raw_x, df_raw_y,
                                                                                      normal_flag=self.train_normal_flag,MinMaxNormalization=self.MinMaxNormalization)
        x_train, y_train, x_val, y_val = self.split_train_val_test(normal_x, normal_y)

        # self.build_scale(x_train)
        self.columns_y = list(df_raw_y.columns)
        self.all_scaler_x = all_scaler_x
        self.all_scaler_y = all_scaler_y
       # self.graph_dict = self.build_graph_data()  # 这里变成标准化后的值，但可以注释代码调整成标准前

        print("x_train, y_train, x_val, y_val: {}, {}, {}, {}".format(x_train.shape,
                                                                      y_train.shape, x_val.shape,
                                                                      y_val.shape))

        self.train_dataloader, self.eval_dataloader = self.gene_dataloader(x_train, y_train, x_val, y_val)

        print("train / val: {}, {}".format(len(self.train_dataloader), len(self.eval_dataloader)))

    def normalization_2D(self, np_data):
        '''
        :param np_data: (N,f)
        :return: (N,f)
        '''

        ls_scaler = []
        for i in range(np_data.shape[1]):
            _scaler = StandardScaler()
            new_np_data = _scaler.fit_transform(np_data[:, i].reshape((-1, 1)))
            ls_scaler.append(_scaler)
            if i == 0:
                all_new_np_data = new_np_data
            else:
                all_new_np_data = np.concatenate((all_new_np_data, new_np_data), axis=1)
        return all_new_np_data, ls_scaler

    def normalization_expext(self, df_data, min_value, max_value, expect_name=None):
        columns = list(df_data.columns)
        if expect_name is not None:
            normal_index = [columns.index(c) for c in columns if c[0] != expect_name]  # index need to normalization
        else:
            normal_index = list(range(len(columns)))
        all_scaler = {}
        for s in normal_index:
            normal_name = columns[s]
            # name = re.findall(r'[a-zA-Z]+', normal_name)[0]

            normal_data = np.array(df_data.iloc[:, s])
            scaler =one_zero_normalization(scale_min=min_value, scale_max=max_value)#这里的df_data确实是包含列名的但是其中dataframe 在传递数据的时候会把数据自动转换成numpy数组
            new_data = scaler.fit_transform(normal_data, v_min=None, v_max=None)  # 将数据归一化至0到1之间
            df_data.loc[:, normal_name] = pd.DataFrame(new_data, columns=[normal_name])
            all_scaler[normal_name] = scaler
        np_data = df_data.values.astype('float32')
        return np_data, all_scaler






    def gene_dataloader(self, x_train, y_train, x_val, y_val):
        train_data = list(zip(x_train, y_train))
        eval_data = list(zip(x_val, y_val))
        print('pad before', len(train_data), len(eval_data))

        if self.pad_with_last_sample:
            num_padding = (self.batch_size - (len(train_data) % self.batch_size)) % self.batch_size
            data_padding = np.repeat(train_data[-1:], num_padding, axis=0)
            train_data = np.concatenate([train_data, data_padding], axis=0)
            num_padding = (self.batch_size - (len(eval_data) % self.batch_size)) % self.batch_size
            data_padding = np.repeat(eval_data[-1:], num_padding, axis=0)
            eval_data = np.concatenate([eval_data, data_padding], axis=0)
        print('pad', len(train_data), len(eval_data))

        train_dataset = ListDataset(train_data)
        eval_dataset = ListDataset(eval_data)

        train_dataloader = DataLoader(dataset=train_dataset, batch_size=self.batch_size,
                                      num_workers=self.num_workers, drop_last=True,
                                      shuffle=True, pin_memory=True)
        eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=self.batch_size,
                                     num_workers=self.num_workers, drop_last=False,
                                     shuffle=False, pin_memory=True)
        return train_dataloader, eval_dataloader

    def generate_input_data(self, df_data_x, df_data_y, normal_flag,MinMaxNormalization):
        """

        Args:
            df_data(pa.DaraFrame): shape: (len_time * 134, feature_dim)

        Returns:
            tuple: tuple contains:
                x(np.ndarray): (size, input_length, 134, feature_dim)
                y(np.ndarray): (size, output_length, 134, feature_dim)
        """
        if add_test_data2train:
            df_train_data = df_data_x[0]
            df_test_data = df_data_x[1]
            all_np_data = pd.concat([df_train_data, df_test_data], ignore_index=True).values.astype('float32')

            if normal_flag == True:
                all_np_data, all_scaler = self.normalization_2D(all_np_data)
            else:
                all_scaler = 0
            np_train_data = all_np_data[0:len(df_train_data), :]
            np_test_data = all_np_data[len(df_train_data):, :]

            x, y = [], []
            x_offsets = np.sort(np.concatenate((np.arange(-self.input_len + 1, 1, 1),)))
            y_offsets = np.sort(np.arange(0, self.output_len, 1))
            min_t = abs(min(x_offsets))

            num_train_samples = np_train_data.shape[0]
            max_t = abs(num_train_samples - abs(max(y_offsets)))
            for t in tqdm(range(min_t, max_t), desc='split data_1'):
                # total = max_t - min_t = n - output_len - input_len + 1
                x_t = np_train_data[t + x_offsets, :]  # [:,6+(-6~0),:]->[:,0~6,0],[:,1~7,:]...
                y_t = np_train_data[t + y_offsets, :]  # [:,6+(1~7),:]->[:,7~13,:],[:,8~14,:]...
                x.append(x_t)  # (node, input_len)
                y.append(y_t)  # (node, input_len)

            num_test_samples = np_test_data.shape[0]
            max_t = abs(num_test_samples - abs(max(y_offsets)))
            for t1 in tqdm(range(min_t, max_t), desc='split data_2'):
                x_t = np_test_data[t1 + x_offsets, :]  # [:,6+(-6~0),:]->[:,0~6,0],[:,1~7,:]...
                y_t = np_test_data[t1 + y_offsets, :]  # [:,6+(1~7),:]->[:,7~13,:],[:,8~14,:]...
                x.append(x_t)  # (node, input_len)
                y.append(y_t)  # (node, input_len)
            print('\nx:', len(x), x[0].shape)
            print('y:', len(y), y[0].shape)
            # x = np.stack(x, axis=0)  # (max_t - min_t, 134, input_len, f)
            # y = np.stack(y, axis=0)  # (max_t - min_t, 134, output_len, f)
            return x, y, all_scaler

        else:
            cols_data_x = df_data_x.columns
            df_data_x = df_data_x[cols_data_x]#到这一步为止他还是一个包含列名的dataframe

            if normal_flag == True:
                # data, all_scaler = self.normalization_2D(data)
                if MinMaxNormalization == True:

                    data_x, all_scaler_x = self.normalization_expext(df_data_x, min_I, max_I)
                    data_y, all_scaler_y = self.normalization_expext(df_data_y, min_P, max_P)
                else:

                    all_scaler_x = StandardScaler()
                    all_scaler_y = StandardScaler()
                    data_x = all_scaler_x.fit_transform(df_data_x).astype('float32')
                    data_y = all_scaler_y.fit_transform(df_data_y).astype('float32')

            else:
                data_x = df_data_x.values.astype('float32')
                data_y = df_data_y.values.astype('float32')
                all_scaler_x = 0
                all_scaler_y = 0

            num_samples = data_x.shape[0]  # t-dim 35280
            # The length of the past time window for the prediction, depends on self.input_length
            x_offsets = np.sort(np.concatenate((np.arange(-self.input_len + 1, 1, 1),)))  # [-6,-5,-4,-3,-2,-1,0] 7size
            # The length of future time window, depends on self.output_length
            y_offsets = np.sort(np.arange(1, self.output_len + 1, 1))  # [1,2,3,4,5,6,7]

            x, y = [], []
            min_t = abs(min(x_offsets))  # input_len-1 143， 7-1=6
            max_t = abs(num_samples - abs(max(y_offsets)))  # n - output_len 35280-288=34992， 10857-7=10850
            # for t in tqdm(range(min_t, max_t,self.output_len), desc='split data'):
            for t in tqdm(range(min_t, max_t), desc='split data'):  # 这里是滑窗 (6,10850),修改成7步
                # total = max_t - min_t = n - output_len - input_len + 1
                x_t = data_x[t + x_offsets, :]  # [:,6+(-6~0),:]->[:,0~6,0],[:,1~7,:]...
                y_t = data_y[t + y_offsets, :]  # [:,6+(1~7),:]->[:,7~13,:],[:,8~14,:]...
                x.append(x_t)  # (node, input_len)
                y.append(y_t)  # (node, input_len)
            print('\nx:', len(x), x[0].shape)
            print('y:', len(y), y[0].shape)
            # x = np.stack(x, axis=0)  # (max_t - min_t, 134, input_len, f)
            # y = np.stack(y, axis=0)  # (max_t - min_t, 134, output_len, f)
            return x, y, all_scaler_x, all_scaler_y

    def split_train_val_test(self, x, y):
        """
        Args:
            x(list): 输入数据 (num_samples, input_len, feature_dim)
            y(list): 输出数据 (num_samples, output_len, feature_dim)

        Returns:
            tuple: tuple contains:
                x_train: (num_samples, input_len, feature_dim)
                y_train: (num_samples, output_len, feature_dim)
                x_val: (num_samples, input_len, feature_dim)
                y_val: (num_samples, output_len, feature_dim)
        """
        unit_x_y_size = len(x) // self.K  # K次均分 11617/5=2323
        board = [0]
        for i in range(1, self.K):
            board.append(board[-1] + unit_x_y_size)
        board.append(len(x))
        print('board:', board)  # [0, 2323, 4646, 6969, 9292, 11617]

        # val
        x_val, y_val = x[board[self.ind]: board[self.ind + 1]], y[board[self.ind]: board[self.ind + 1]]
        print('val范围：', board[self.ind], ':', board[self.ind + 1])
        # train
        x_train, y_train = [], []
        for i in range(self.K):  # （0，5）
            # if i == self.ind:
            #      continue  # 排除已经被选为验证集的部分
            print('train范围：', board[i], ':', board[i + 1])
            x_i = x[board[i]: board[i + 1]]
            y_i = y[board[i]: board[i + 1]]
            x_train += x_i
            y_train += y_i
        print('length->', 'x_train:', len(x_train), 'y_train:', len(y_train), 'x_val:', len(x_val), 'y_val', len(y_val))
        x_train = np.array(x_train)  # (b, n, f)
        y_train = np.array(y_train)
        x_val = np.array(x_val)
        y_val = np.array(y_val)
        print('shape->', 'x_train:', x_train.shape, 'y_train:', y_train.shape, 'x_val:', x_val.shape, 'y_val:',
              y_val.shape)
        return x_train, y_train, x_val, y_val

    def build_graph_data(self):
        graph_dict = dict()
        geo_graph = pd.read_excel(os.path.join(self.graph_path, self.graph_file[0]))
        geo_graph = geo_graph.iloc[:self.capacity, :self.capacity]
        graph_dict['graph_1'] = geo_graph
        return graph_dict

    @staticmethod
    def test_data_upsample():
        # 取测试集数据的前5%进行上采样，并加入到训练
        up_sample_bata = 5  # 上采样倍数
        df_data = pd.read_csv(r"./TM_data/data/TM_2UNIT_test_change.csv")  # 2w个点
        df_data = df_data.dropna()
        df_data = df_data.iloc[0:int(len(df_data) * 0.05), 1:25]  # 取前5%（1k个点）
        np_data = np.array(df_data)
        l, n = np_data.shape
        x = np.linspace(1, l, l)
        new_x = np.linspace(1, l, up_sample_bata * l)
        new_data = np.zeros((up_sample_bata * l, n))

        for i in range(n):
            interp = interp1d(x, np_data[:, i], kind='linear')
            new_data[:, i] = interp(new_x)
            # # plot result
            # plt.figure()
            # plt.plot(x, np_data[:, i], c='r', label='original')
            # plt.plot(new_x, new_data[:, i], c='b', label='upsample')
            # plt.title(f'I_{i + 1}')
            # plt.legend()
            # plt.show()
            # plt.close()
            # plt.clf()
        return new_data

    def upsample(self, data, rate):
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


class Test_I_Dataset(Dataset):

    def __init__(self, filename, capacity, normal_flag, all_scaler_x, all_scaler_y, Num_sample, check_flag=False, MinMaxNormalization=False,
                 FD_flag=False,
                 range_check_threshold=None):

        super().__init__()
        self.filename = filename
        self.capacity = capacity
        self.normal_flag = normal_flag
        self.all_scaler_x = all_scaler_x
        self.all_scaler_y = all_scaler_y
        self.N_sample = Num_sample
        self.check_flag = check_flag
        self.MinMaxNormalization = MinMaxNormalization

        self.start_col = 0
        self._logger = getLogger()
        self.FD = FD_flag
        self.range_check_threshold = range_check_threshold

        self.__read_data__()

    def __read_data__(self):
        df = pd.read_csv(self.filename)
        df_data_x = df.iloc[:, 1:7 + 1]
        # df_data_x = df.iloc[:, 24 + 1 + 4:24 + 1 + 4 + 24]

        # df_raw_I = df.iloc[:, 1:24 + 1]  # 第一列是time
        # df_raw_P = df.iloc[:, 24 + 1 + 4:24 + 1 + 4 + 24]
        # df_data_x = pd.concat([df_raw_I, df_raw_P], axis=1)

        df_data_y = df.iloc[:, 1:7 + 1]
        df_data_x = df_data_x.dropna()
        df_data_y = df_data_y.dropna()

        # down_simple = np.arange(0, len(df_data), 1)  # 对测试集进行下采样
        # df_data = df_data.iloc[down_simple, :]
        # self.num_testX = df_data.shape[0]
        # df_data = self.check_change_rage(df_data)
        self.df_data_x = df_data_x
        self.df_data_y = df_data_y

        data_x = self.build_graph_data(df_data_x, self.normal_flag, self.all_scaler_x,MinMaxNormalization=self.MinMaxNormalization)  # 这里的数据是一个包含列名的dataframe
        data_y = self.build_graph_data(df_data_y, self.normal_flag, self.all_scaler_y,MinMaxNormalization=self.MinMaxNormalization)  # (1,N,T,F)
        self.data_x = data_x
        self.data_y = data_y

    def get_raw_df(self):
        return self.df_data

    def Testnormalization_2D(self, np_data):
        '''
        :param np_data: (N, f)
        :return: (N, f)
        '''
        for i in range(np_data.shape[1]):
            _scaler = self.all_scaler[i]
            temp_np = np_data[:, i].reshape((-1, 1))
            new_np_data = _scaler.transform(temp_np)
            if i == 0:
                all_new_np_data = new_np_data
            else:
                all_new_np_data = np.concatenate((all_new_np_data, new_np_data), axis=1)
        return all_new_np_data

    def normalization(self, df_data, all_scaler,MinMaxNormalization):#这里传进来的df_data是dataframe
        if MinMaxNormalization == True:

            columns = list(df_data.columns)
            self.columns = columns
            expect_I_index = [columns.index(c) for c in columns if c[0] != 'I']
            for s in expect_I_index:
                expect_I_name = columns[s]
                scaler = all_scaler[expect_I_name]

                expect_I_data = np.array(df_data.iloc[:, s])
                new_data = scaler.transform(expect_I_data)
                df_data.loc[:, expect_I_name] = pd.DataFrame(new_data, columns=[expect_I_name], index=df_data.index)
                np_data = df_data.values.astype('float32')
        else:
            scaler = all_scaler
            np_data = scaler.transform(df_data).astype('float32')


        return np_data

    def build_graph_data(self, df_data, normal_flag, all_scaler,MinMaxNormalization):#这里的df_data是一个包含列名的dataframe
        cols_data = df_data.columns
        df_data_min_max_norm = df_data[cols_data]
        data = df_data_min_max_norm.values

        if normal_flag == True:

            data = self.normalization(df_data_min_max_norm , all_scaler, MinMaxNormalization)#这里的data是numpy数组


        return np.expand_dims(data, [0])

    def get_data_x(self):
        return self.data_x

    def get_data_y(self):
        return self.data_y


if __name__ == '__main__':
    I_Dataset.test_data_upsample()
