from torch.utils.data import Dataset, DataLoader
import numpy as np


class ListDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def dataset(data, input_len, pred_len, step, bs, shuffle=True):
    X = []
    Y = []
    for i in range(0, data.shape[0], step):
        if (i + input_len + pred_len) > data.shape[0]:
            break
        x = data[i:i + input_len, :]
        y = data[i + input_len:i + input_len + pred_len, :]
        X.append(x[np.newaxis, ...])
        Y.append(y[np.newaxis, ...])
    X = np.concatenate(X, axis=0).astype(np.float32)
    Y = np.concatenate(Y, axis=0).astype(np.float32)
    data = list(zip(X, Y))
    dataset = ListDataset(data)
    dataloader = DataLoader(dataset=dataset, batch_size=bs,
                            num_workers=0, drop_last=True,
                            shuffle=shuffle, pin_memory=True)
    return dataloader


def ML_dataset(data, input_len, pred_len, step=1):
    """
    :param data:[len, node]
    :param input_len:
    :param pred_len:
    :param step:
    :return: X,Y:[bs, input_len, node],[bs, pred_len, node]
    """
    X = []
    Y = []
    for i in range(0, data.shape[0], step):
        if (i + input_len + pred_len) > data.shape[0]:
            break
        x = data[i:i + input_len, :]
        y = data[i + input_len:i + input_len + pred_len, :]
        X.append(x[np.newaxis, ...])
        Y.append(y[np.newaxis, ...])
    X = np.concatenate(X, axis=0).astype(np.float32)
    Y = np.concatenate(Y, axis=0).astype(np.float32)
    return X, Y
