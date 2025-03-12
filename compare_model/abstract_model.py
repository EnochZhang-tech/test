import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import os
from copy import deepcopy
import pickle
from compare_model.utils import FilterHuberLoss


# father of ML model and DL model
class Deep_Learn():
    def __init__(self, config, epochs, lr, patient, device, out_path):
        self.config = config
        self.device = device
        self.epochs = epochs
        self.lr = lr
        self.patient = patient
        self.loss_fn = FilterHuberLoss()
        self.out_path = out_path

    def train_model(self, model, dataloader, node: int):
        """
        train for one node
        :param model:
        :param dataloader:
        :param node:
        :return:
        """
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        opt.zero_grad()
        all_pred_loss = []
        best_score = np.inf
        patient = 0
        best_id = -1
        model.to(self.device)
        model.train()
        for epoch in range(self.epochs):
            pred_losses = []
            for data, true in dataloader:
                # data = torch.cat([data[..., 0:node], data[..., node + 1:]], dim=-1).to(self.device)
                data = data[..., node].to(self.device)
                true = true[..., node].to(self.device)
                pred = model(data)  # [bs, pred_len]
                loss = self.loss_fn(pred, true)

                opt.zero_grad()
                loss.backward()
                opt.step()
                pred_losses.append(loss.item())

            pred_loss = sum(pred_losses) / len(pred_losses)
            print(f"Model={node}, Epoch={epoch}, seq_Loss: {pred_loss:.7f}")
            all_pred_loss.append(pred_loss)
            best_score = min(pred_loss, best_score)
            if best_score == pred_loss:
                patient = 0
                best_id = epoch
                self.save_model(self.out_path, node, model, steps=epoch)
                print("---Saving the current model---")
            else:
                patient += 1
                if patient > self.patient:
                    print("----Model Patient-----Earlystopping")
                    break
        return best_id

    def predict_model(self, model, test_dataloader, node):
        model.to(self.device)
        model.eval()
        all_pred = []
        all_gt = []
        for data, true in test_dataloader:
            # data = torch.cat([data[..., 0:node], data[..., node + 1:]], dim=-1).to(self.device)
            data = data[..., node].to(self.device)
            true = true[..., node]
            pred = model(data)

            pred = np.array(pred.cpu().detach()).reshape((-1, 1))
            true = np.array(true.cpu().detach()).reshape((-1, 1))
            all_pred.append(pred)
            all_gt.append(true)
        all_pred = np.concatenate(all_pred, axis=0)
        all_gt = np.concatenate(all_gt, axis=0)
        return all_pred, all_gt

    def fit(self, train_dataloader, model_class):
        num_node = self.num_node
        self.all_best_id = []
        for node in tqdm(range(num_node), total=num_node, desc='train'):
            model = model_class(self.config)
            best_id = self.train_model(model, train_dataloader, node)
            self.all_best_id.append(best_id)

    def predict(self, test_dataloader, model_class):
        if not hasattr(self, 'all_best_id'):
            self.all_best_id = []
            for i in range(self.num_node):
                model_path = os.listdir(os.path.join(self.out_path, f'node_{i}'))
                id = [int(name.split('.')[0].split('_')[1]) for name in model_path]
                self.all_best_id.append(max(id))
        model = model_class(self.config)
        all_pred = []
        all_gt = []
        for node in tqdm(range(self.num_node), total=self.num_node, desc='test'):
            best_metric = np.inf
            # for best in range(self.config.epochs):
            for best in range(self.all_best_id[node], self.all_best_id[node] + 1):
                try:
                    self.load_model(output_path=self.out_path, node=node, best_id=best, model=model)
                except:
                    continue
                pred, gt = self.predict_model(model, test_dataloader, node)
                smape = self.smape(pred, gt)
                if smape < best_metric:
                    best_pred = pred
                    best_gt = gt
                    best_metric = smape
            all_pred.append(best_pred)
            all_gt.append(best_gt)
        all_pred = np.concatenate(all_pred, axis=-1)
        all_gt = np.concatenate(all_gt, axis=-1)
        return all_pred, all_gt

    def save_model(self, output_path, node, model, steps=None):
        output_path = os.path.join(output_path, f'node_{node}')
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        output_dir = os.path.join(output_path, "model_%d.pt" % steps)
        torch.save(model.state_dict(), output_dir)

    def load_model(self, output_path, node, best_id, model):
        model_state = torch.load(os.path.join(output_path, f"node_{node}/model_{best_id}.pt"), map_location='cuda')
        model.load_state_dict(model_state)

    def smape(self, pred, gt):
        _smape = []
        for node in range(pred.shape[1]):
            node_pred = pred[:, node]
            node_gt = gt[:, node]
            if len(node_pred) > 0 and len(node_gt) > 0:
                _smape.append(2.0 * np.mean(np.abs(node_pred - node_gt) / (np.abs(node_pred) + np.abs(node_gt))) * 100)
                # _smape.append(np.abs(node_pred - node_gt))
        return np.mean(np.array(_smape))


class Machine_Learn():
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.out_path = config.out_path
        self.input_len = config.input_len
        self.pred_len = config.pred_len
        self.direct_predict = config.direct_predict_multi_step

    def fit_model(self, x, y, _model):
        if self.direct_predict:
            self.fit_model_direct(x, y, _model)
        else:
            self.fit_model_iter(x, y, _model)

    def predict(self, test_x):
        if self.direct_predict:
            all_pred = self.predict_direct(test_x)
        else:
            all_pred = self.predict_iter(test_x)
        return all_pred

    """直接多步预测"""

    def fit_model_direct(self, x, y, _model):
        """
        :param x:[bs, 45, 4]
        :param y:[bs, 15, 4]
        """
        for n in tqdm(range(x.shape[-1]), total=x.shape[-1], desc='train'):

            model = deepcopy(_model)
            train_x = x[..., n]
            train_y = y[..., n]
            model.fit(train_x, train_y)
            self.save_model(model, self.out_path, model_index=n)

    def predict_direct(self, test_x):
        all_node_pred = []
        for n in tqdm(range(test_x.shape[-1]), total=test_x.shape[-1], desc='predict'):
            x = test_x[..., n]
            model = self.load_model(self.out_path, model_index=n)
            one_node_pred = []
            for sample in range(x.shape[0]):
                _x = x[sample, :].reshape((1, -1))  # [1, in_len]
                pred = model.predict(_x).reshape((-1, 1))
                one_node_pred.append(pred)
            all_node_pred.append(np.concatenate(one_node_pred, axis=0))
        all_node_pred = np.concatenate(all_node_pred, axis=-1)
        return all_node_pred

    """迭代多步预测"""

    def fit_model_iter(self, x, y, _model):
        """
        :param x:[bs, in_len, 24]
        :param y:[bs, 1, 24]
        """
        for n in tqdm(range(x.shape[-1]), total=x.shape[-1], desc='train'):
            model = deepcopy(_model)
            train_x = x[..., n]
            train_y = y[:, 0, n]
            model.fit(train_x, train_y)
            self.save_model(model, self.out_path, model_index=n)

    def predict_iter(self, test_x):
        all_node_pred = []
        for n in tqdm(range(test_x.shape[-1]), total=test_x.shape[-1], desc='predict'):
            x = test_x[..., n]
            model = self.load_model(self.out_path, model_index=n)
            one_node_pred = []
            for sample in range(x.shape[0]):
                _x = x[sample, :].reshape((1, -1))  # [1, in_len]
                pred = []
                for i in range(self.pred_len):  # 预测pred_len步
                    y = model.predict(_x).reshape((-1, 1))  # [1, 1]
                    pred.append(y)
                    _x = np.concatenate([_x[:, 1:], y], axis=-1)
                one_node_pred.append(np.concatenate(pred, axis=0))
            all_node_pred.append(np.concatenate(one_node_pred, axis=0))
        all_node_pred = np.concatenate(all_node_pred, axis=-1)
        return all_node_pred

    def save_model(self, model, path, model_index):
        if not os.path.exists(path):
            os.makedirs(path)
        with open(path + f'/model_{model_index + 1}.pkl', 'wb') as file:
            pickle.dump(model, file)

    def load_model(self, path, model_index):
        with open(path + f'/model_{model_index + 1}.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
