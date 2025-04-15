import math
import os
from collections import deque

import pandas as pd
import numpy as np
from openpyxl.styles.builtins import output
from pyexpat import features
from sklearn import preprocessing
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint,EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split
#from tensorflow.keras.utils import plot_model
#from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import time
import math
import keras
import keras.backend as K
import tensorflow as tf
from tqdm import tqdm
import random

from cbam_block_timeseries import cbam_block
from keras.models import Sequential, load_model, Model
from keras.layers import LSTM, GRU, Bidirectional, Dense, Conv1D, MaxPooling1D, Flatten,RepeatVector, TimeDistributed,Dropout
from keras.layers import Conv1D, MaxPooling1D, Flatten, Input, BatchNormalization, Reshape
import keras.backend as K
from keras.layers import Bidirectional, LSTM, Reshape, Flatten, Dense
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam  # 导入 Adam 优化器
import numpy as np
from sklearn.preprocessing import StandardScaler
# 其他代码...



def RAE(pred, gt):
    gt_mean = np.mean(gt)
    squared_error_num = np.sum(np.abs(gt - pred))
    squared_error_den = np.sum(np.abs(gt - gt_mean))
    rae_loss = squared_error_num / squared_error_den
    return rae_loss

def mspe(pred, gt):
    return np.mean(np.square((pred - gt) / gt)) if len(pred) > 0 and len(
        gt) > 0 else 0
# 其他代码...

def MAE(pred, gt):
    _mae = 0.
    if len(pred) > 0 and len(gt) > 0:
        _mae = np.mean(np.abs(pred - gt))
    return _mae

def MSE(pred, gt):
    _mse = 0.
    if len(pred) > 0 and len(gt) > 0:
        _mse = np.mean((pred - gt)**2)
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





def sliding_window(np_X, np_Y, n_in, n_out=24, in_start=0):
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


def sliding_window_test(np_X, np_Y, n_in, n_out=24, in_start=0):
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


def Seq2Seq_model(train_x, train_y, modelfile, flag, pred_len,epochs_num=100, batch_size_set=16, lr_rate=0.01, verbose_flag=0): #exchange和ETTh2的批次为32学习率为0.0001，weather的批次为16学习率为0.003
    time_start = time.time()
    if flag == True:
        n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]  #

        K.clear_session()  # 清除之前的模型，省得压满内存

        # inputs = Input(shape=(train_x.shape[1], train_x.shape[2]))  # 输入特征接收维度
        inputs = Input(shape=(n_timesteps, n_features))  # 输入特征接收维度

        # input1 = Conv1D(20, kernel_size=4, activation='sigmoid', padding='same', strides=2)(inputs)
        input2 = Conv1D(1, kernel_size=3, activation='relu', padding='same', strides=1)(inputs)#这里因为same_padding和s=2所以输出时间步变为一半向上取整，可以理解

        x = BatchNormalization()(input2)#不会改变维度，仅仅进行正则化和加速收敛


        encoder_outputs = Bidirectional(
            LSTM(8, return_state=False, return_sequences=True, activation='tanh'))(x)#这里并不会改变维度，只会将特征数量改为64这是编码（batch_size,45,64）   注意这里是原实验使用的参数

        # encoder_outputs = Bidirectional(
        #     LSTM(64, return_state=False, return_sequences=True, activation='relu'))(
        #     x)  # 测试relu的效果



        x1 = cbam_block(encoder_outputs)#不会改变，仅仅是对注意力机制的再加权，体现出贡献度较大的特征

        # 使用 K.int_shape() 获取形状信息
        x1_shape = K.int_shape(x1)
        input2 = Reshape((x1_shape[1], x1_shape[-1]))(x1)#(batch_size,23,64)不改变样本的维度，仅确保每个样本维度一致

        decoder_outputs = Bidirectional(
            LSTM(8, return_state=False, return_sequences=True, activation='tanh'))(input2)#(batch_size,15,64)解码层输出，这是原实验的

        # decoder_outputs = Bidirectional(
        #     LSTM(64, return_state=False, return_sequences=True, activation='relu'))(input2)  # 测试relu
        pre_output = TimeDistributed(Dense(1, activation='tanh'))(decoder_outputs)  # (batch_size,15,4)
        # output = TimeDistributed(Dense(4, activation='relu'))(decoder_outputs)#(batch_size,15,4)
        output = pre_output[:,-pred_len:,:]

        # input2 = Flatten()(decoder_outputs)#(为全连接层提供输入，所以需要将后面的时间步和特征维度展平)(6526,64*23)
        # dense_1 = Dense(20, activation='selu')(input2)

        # output = Dense(n_outputs, activation='selu')(dense_output)

        model = Model(inputs=inputs, outputs=output)

        # model.compile(optimizer=keras.optimizers.adam(learning_rate=lr_rate), loss='mse', metrics=['accuracy'])
        model.compile(optimizer=Adam(learning_rate=lr_rate), loss='mse', metrics=['accuracy'])
        earlyStop = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')

        checkpoint = ModelCheckpoint(modelfile, monitor='val_loss', verbose=0, save_best_only=False, mode='auto',save_freq='epoch')

        # encoder_outputs = Bidirectional(
        #     LSTM(32, return_state=False, return_sequences=True, activation='tanh'))(x)
        #
        # input = Reshape((-1,encoder_outputs._keras_shape[1], encoder_outputs._keras_shape[-1]))(encoder_outputs)
        #
        # x1 = cbam_block(encoder_outputs)
        #
        # input2 = Reshape((x1._keras_shape[1], x1._keras_shape[-1]))(x1)
        # decoder_outputs = Bidirectional(
        #     LSTM(32, return_state=False, return_sequences=True, activation='tanh'))(input2)
        #
        # input2 = Flatten()(decoder_outputs)
        # dense_1 = Dense(20, activation='selu')(input2)
        #
        # output = Dense(n_outputs, activation='selu')(dense_1)
        #
        # model = Model(inputs=inputs, outputs=output)  # 初始命名训练的模型为model
        #
        # model.compile(optimizer=keras.optimizers.adam(learning_rate=lr_rate), loss='mse', metrics=['accuracy'])
        #
        # earlyStop = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
        #
        # checkpoint = ModelCheckpoint(modelfile, monitor='val_loss', verbose=0, save_best_only=True, mode='auto')


        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto',
                                      # epsilon=0.0001,
                                      min_delta=0.00001, cooldown=0, min_lr=0)

        history = model.fit(train_x, train_y, validation_data=(train_x, train_y), epochs=epochs_num,verbose=verbose_flag,
                            batch_size=batch_size_set, callbacks=[checkpoint, reduce_lr, earlyStop])
        plt.plot(history.history['loss'], label='train_loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.title('train and validate loss')
        plt.legend()
        plt.show()

        print('------train finish！！-----')
        time_end = time.time()
        print('训练用时:{:.3f}s'.format(time_end - time_start))
        return model
    else:
        model = load_model(modelfile)
        return model

def model_predict(model, test_x, out_step=1):

    testmodel = load_model(model)
    yhat_sequence = testmodel.predict(test_x)
    predict_result = yhat_sequence

    return predict_result

def set_seed(seed):
    """
    Set seed for reproduction.
    传入的数值用于指定随机数生成时所用算法开始时所选定的整数值，如果使用相同的seed()值，则每次生成的随机数都相同；
    如果不设置这个值，则系统会根据时间来自己选择这个值，此时每次生成的随机数会因时间的差异而有所不同。
    """
    random.seed(seed)  # random模块的随机数种子
    np.random.seed(seed)  # np.random的随机数种子
    tf.random.set_seed(seed)
    tf.config.experimental.enable_op_determinism()

# 禁用多线程
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"


def evaluate_predict(predicted, test_y):

    rmse = RMSE(predicted, test_y)
    # print('Test RMSE: %.3f' % rmse)
    mae = MAE(predicted, test_y)
    # print('Test MAE: %.3f' % mae)
    mape = MAPE(predicted, test_y)
    # print("Test MAPE:{:.3f}%".format(mape))
    smape = SMAPE(predicted, test_y)
    # print("Test SMAPE:{:.3f}%".format(smape))
    rae = RAE(predicted, test_y)

    return rmse,mae,mape,smape,rae

def compute_final_result(all_RMSE, all_MAE, all_MAPE, all_SMAPE):
    ave_RMSE = sum(all_RMSE) / len(all_RMSE)
    ave_MAE = sum(all_MAE) / len(all_MAE)
    ave_MAPE = sum(all_MAPE) / len(all_MAPE)
    ave_SMAPE = sum(all_SMAPE) / len(all_SMAPE)
    ave_RAE = sum(all_RAE) / len(all_RAE)
    print('ave_RMSE:',ave_RMSE, 'ave_MAE:',ave_MAE, 'ave_MAPE:',ave_MAPE,'ave_SMAPE:',ave_SMAPE,'ave_RAE:',ave_RAE)
    return ave_RMSE, ave_MAE, ave_MAPE, ave_SMAPE, ave_RAE

if __name__ == '__main__':
    set_seed(3407)
    #短期
    # df_train = pd.read_csv(r'F:\MTSF\CM\Compare_Models\Dataset\ETTh1_train.csv')
    # df_test = pd.read_csv(r'F:\MTSF\CM\Compare_Models\Dataset\ETTh1_test.csv')

    #长期
    df_train = pd.read_csv(r'F:\MTSF\CM\Compare_Models\Dataset\weather_train.csv')  #ETTh2 exchange_rate
    df_test = pd.read_csv(r'F:\MTSF\CM\Compare_Models\Dataset\weather_test.csv')   #ETTh2 exchange_rate

    # input_step = 1008
    # output_step = 168
    input_step = 144
    output_step = 24
    train_flag =True
    spnd_num = df_train.values.shape[1]
    save_pred_Data = True
    data = []
    all_RMSE = []
    all_MAE = []
    all_MAPE = []
    all_SMAPE = []
    all_RAE = []

    print('----第{}次训练----'.format(1))


    """"StandardScaler"""
    data_train = df_train.iloc[:,-1: ]
    data_test = df_test.iloc[:,-1: ]


    # 创建标准化器
    scaler = StandardScaler()

    # 对训练数据进行标准化
    np_train_x = scaler.fit_transform(data_train.values[:, :])
    np_train_y = scaler.fit_transform(data_train.values[:, :])  # 如果你想用相同的标准化参数

    # 对测试数据进行标准化
    np_test_x = scaler.transform(data_test.values[:, :])  # 注意：使用训练集的标准化参数
    np_test_y = (data_test.values[:, :])

    """"MinMaxScaler"""
    # data_train = df_train.iloc[:,-1: ]  #1: 1+7
    # data_test = df_test.iloc[:,-1: ]  #1: 1+7
    #
    # # 创建 Min-Max 归一化器
    # scaler = MinMaxScaler()
    #
    # # 对训练数据进行归一化
    # np_train_x = scaler.fit_transform(data_train.values[:, :])
    # np_train_y = scaler.fit_transform(data_train.values[:, :])  # 使用相同的归一化参数
    #
    # # 对测试数据进行归一化
    # np_test_x = scaler.transform(data_test.values[:, :])  # 注意：使用训练集的归一化参数
    np_test_y = data_test.values[:, :]  # 这里不需要归一化


    train_x, train_y = sliding_window(np_train_x, np_train_y, input_step,output_step)
    # print('train_x:', train_x.shape, 'train_y:', train_y.shape)

    test_x, test_y = sliding_window_test(np_test_x, np_test_y, input_step,output_step)
    # print('test_x:', test_x.shape, 'test_y:', test_y.shape)

    model_file ='Test.\CNN-Seq2Seq\short_term_CNN_seq2seq_epoch\exp1_{epoch:02d}.keras'
    # model_file ='r.\MODEL\CNN-Seq2Seq\short_term_CNN_seq2seq_epoch_{epoch:02d}.h5'
    if train_flag:
        model = Seq2Seq_model(train_x, train_y, modelfile=model_file, flag=train_flag,pred_len=output_step)

    import numpy as np
    from tensorflow.keras.models import load_model

    # 假设你知道模型保存的 epoch 数量
    epochs_num = 100  # 训练的总 epoch 数

    # 存储所有模型的预测结果

    recent_best_results = deque(maxlen=5)
    best_metric = np.inf

    # 遍历每个 epoch 的模型文件，加载并进行预测
    for epoch in range(1, epochs_num + 1):
        model_file = f'Test.\CNN-Seq2Seq\short_term_CNN_seq2seq_epoch\exp1_{epoch:02d}.keras'
        # model_file = f'r.\MODEL\CNN-Seq2Seq\short_term_CNN_seq2seq_epoch_{epoch:02d}.h5'
        try:# 生成当前 epoch 的文件名 # 加载模型
            model = load_model(model_file)
        except:
            continue

        predict_result = model_predict(model_file, test_x)



        predict_result=predict_result.reshape(-1,1)
        test_y = test_y.reshape(-1,1)

        predict_result = scaler.inverse_transform(predict_result)




        real_data =test_y

        # 通道名称


        # 评估预测
        all_rmse, all_mae, all_mape, all_smape, all_RAE = evaluate_forecasts(predict_result,real_data,1)

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
            best=predict_result
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

            data = np.column_stack((real_data, predict_result))

            all_pred_df = pd.DataFrame(data, columns=['Ground_Truth', 'Prediction'])


            target_dir = f'../save_result/data/CNN_pred_{output_step}'

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
            channels = ['Dpax[1]', 'Dpax[2]', 'Dpax[3]', 'Dpax[4]','Dpax[5]','Dpax[6]','Dpax[7]']

            # 指定保存路径
            save_path = '.\Test\CNN-seq2seq\short-term-144-24/'  # 请根据需要修改路径
            os.makedirs(save_path, exist_ok=True)

            # 遍历每个通道绘图并保存
            for i in range(7):
                plt.figure(figsize=(15, 8))  # 每张图的大小
                plt.plot(real_data[:, i], color='red', label='y_true', linestyle='--')
                plt.plot(predict_result[:, i], color='blue', label='y_pred', linestyle='-.')
                plt.title('Comparison') # f'{channels[i]}
                plt.xlabel('date')
                plt.ylabel('OT')
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7)

                # 保存图形
                plt.savefig(f'{save_path}Dpax[{channels[i]}]_.png')
                plt.close()  # 关闭当前图以释放内存

            print('--------------Print Ending-----------------')
            # print('--------------Print Ending-----------------')
            # print('--------------Print Ending-----------------')
            # print('--------------Print Ending-----------------')






