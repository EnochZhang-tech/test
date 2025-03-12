# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys
import os
import logging
import datetime
import torch
import argparse
import numpy as np
import pandas as pd

def build_optimizer(config, log, model):
    log.info('You select `{}` optimizer.'.format(config.learner.lower()))
    if config.learner.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    elif config.learner.lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=config.lr,
                                    momentum=config.lr_momentum, weight_decay=config.weight_decay)
    elif config.learner.lower() == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=config.lr,
                                        eps=config.lr_epsilon, weight_decay=config.weight_decay)
    elif config.learner.lower() == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=config.lr,
                                        alpha=config.lr_alpha, eps=config.lr_epsilon,
                                        momentum=config.lr_momentum, weight_decay=config.weight_decay)
    elif config.learner.lower() == 'sparse_adam':
        optimizer = torch.optim.SparseAdam(model.parameters(), lr=config.lr,
                                           eps=config.lr_epsilon, betas=config.lr_betas)
    elif config.learner.lower() == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    else:
        config.log.warning('Received unrecognized optimizer, set default Adam optimizer')
        optimizer = torch.optim.Adam(config.model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    return optimizer


def build_lr_scheduler(config, log, optimizer):
    """
    select lr_scheduler
    """
    if config.lr_decay:
        log.info('You select `{}` lr_scheduler.'.format(config.lr_scheduler_type.lower()))
        if config.lr_scheduler_type.lower() == 'multisteplr':
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=config.milestones, gamma=config.lr_decay_ratio)
        elif config.lr_scheduler_type.lower() == 'steplr':
            lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=config.step_size, gamma=config.lr_decay_ratio)
        elif config.lr_scheduler_type.lower() == 'exponentiallr':
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=config.lr_decay_ratio)
        elif config.lr_scheduler_type.lower() == 'cosineannealinglr':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config.lr_T_max, eta_min=config.lr_eta_min)
        elif config.lr_scheduler_type.lower() == 'lambdalr':
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=config.lr_lambda)
        elif config.lr_scheduler_type.lower() == 'reducelronplateau':
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=config.lr_patience,
                factor=config.lr_decay_ratio, threshold=config.lr_threshold)
        else:
            log.warning('Received unrecognized lr_scheduler, please check the parameter `lr_scheduler`.')
            lr_scheduler = None
    else:
        lr_scheduler = None
    return lr_scheduler

def data_augment(X, y, p=0.8, alpha=0.5, beta=0.5):
    """

    Args:
        X: B,N,T
        y: B,N,T
        p:
        alpha:
        beta:

    Returns:

    """
    """Regression SMOTE
    """
    batch_size = X.shape[0]  # 32
    random_values = torch.rand([batch_size])  # size[32] 随机数
    idx_to_change = random_values < p  # size[32] 转成True or False

    # ensure that first element to switch has probability > 0.5
    np_betas = np.random.beta(alpha, beta, batch_size) / 2 + 0.5  # np size[32] β
    random_betas = torch.FloatTensor(np_betas).reshape([-1, 1, 1])  # tensor [32,1,1,1] β
    index_permute = torch.randperm(batch_size)  # size[32] 将0~batch_size-1（包括0和batch_size-1）随机打乱后获得的数字序列

    X[idx_to_change] = random_betas[idx_to_change] * X[idx_to_change]
    X[idx_to_change] += (1 - random_betas[idx_to_change]) * X[index_permute][idx_to_change]

    y[idx_to_change] = random_betas[idx_to_change] * y[idx_to_change]
    y[idx_to_change] += (1 - random_betas[idx_to_change]) * y[index_permute][idx_to_change]
    return X, y


def save_model(output_path,
               model,
               steps=None):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_dir = os.path.join(output_path, "model_%d.pt" % steps)
    torch.save(model.state_dict(), output_dir)

def load_model(output_path, model, opt=None, lr_scheduler=None, log=None):
    if log is not None:
        log.info("load model from  %s" % output_path)
    # model_state = torch.load(output_path)
    model_state = torch.load(output_path, map_location='cuda')
    model.load_state_dict(model_state)

def str2bool(s):
    if isinstance(s, bool):
        return s
    if s.lower() in ('yes', 'true'):
        return True
    elif s.lower() in ('no', 'false'):
        return False
    else:
        raise argparse.ArgumentTypeError('bool value expected.')


def _create_if_not_exist(path):
    basedir = os.path.dirname(path)
    if not os.path.exists(basedir):
        os.makedirs(basedir)


def get_local_time():
    cur = datetime.datetime.now()
    cur = cur.strftime('%b-%d-%Y_%H-%M-%S')
    return cur


def get_logger(config, name=None):
    """
    Logger

    Args:
        config(ConfigParser): config
        name: specified name

    Returns:
        Logger: logger
    """
    log_dir = './log'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_filename = '{}-{}-{}-{}.log'.format(config.exp_id, config.model, config.filename[:-4], get_local_time())
    logfilepath = os.path.join(log_dir, log_filename)

    logger = logging.getLogger(name)

    log_level = config.get('log_level', 'INFO')

    if log_level.lower() == 'info':
        level = logging.INFO
    elif log_level.lower() == 'debug':
        level = logging.DEBUG
    elif log_level.lower() == 'error':
        level = logging.ERROR
    elif log_level.lower() == 'warning':
        level = logging.WARNING
    elif log_level.lower() == 'critical':
        level = logging.CRITICAL
    else:
        level = logging.INFO

    logger.setLevel(level)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(logfilepath)
    file_handler.setFormatter(formatter)

    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info('Log directory: %s', log_dir)
    return logger


def ensure_dir(dir_path):
    """Make sure the directory exists, if it does not exist, create it.

    Args:
        dir_path (str): directory path
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


class one_zero_normalization():
    def __init__(self, scale_min=0, scale_max=1):
        """
        scale to the value in [scale_min, scale_max]
        """
        assert scale_max > scale_min, "scale_max must bigger than scale_min"
        self.scale_min = scale_min
        self.scale_max = scale_max

    def fit(self, np_data, v_min, v_max):  # 1d data
        if v_max and v_min:
            self.v_min = v_min
            self.v_max = v_max
        else:
            self.v_min = np.min(np_data)
            self.v_max = np.max(np_data)


    def transform(self, np_data):  # 1d data
        new_data = self.scale_min + ((np_data - self.v_min) / (self.v_max - self.v_min)) * (
                self.scale_max - self.scale_min)
        return new_data

    def fit_transform(self, np_data, v_min, v_max):
        self.fit(np_data, v_min, v_max)
        return self.transform(np_data)

    def inverse_transform(self, np_data):  # 1d data
        new_data = self.v_min + ((np_data - self.scale_min) / (self.scale_max - self.scale_min)) * (
                self.v_max - self.v_min)
        return new_data





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

    try:
        save_df.to_csv(os.path.join(file_path, f'{file_name}.csv'), index=False)
    except Exception as e:
        print(f"Error saving CSV file: {e}")
