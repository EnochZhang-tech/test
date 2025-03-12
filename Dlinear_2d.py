import torch
from torch import nn

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool2d(kernel_size=(1, kernel_size), stride=stride, padding=0)

    def forward(self, x):
        # x: [Batch, in_dim, node, Input_length]
        # padding on the both ends of time series
        front = x[..., 0:1].repeat(1, 1, 1, (self.kernel_size - 1) // 2)
        end = x[..., -1:].repeat(1, 1, 1, (self.kernel_size - 1) // 2)
        x = torch.cat([front, x, end], dim=-1)
        x = self.avg(x)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        # x: [Batch, in_dim, node, Input_length]
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class DLinear(nn.Module):
    def __init__(self, config,  ks=7, individual=None):#adj_mx,
        super(DLinear, self).__init__()
        self.Lag = config["input_len"]
        self.Horizon = config["output_len"]
        self.out_nodes = config['out_capacity']
        self.individual = individual
        in_dim = config["var_len"]

        # Decompsition Kernel Size
        kernel_size = ks
        self.decompsition = series_decomp(kernel_size)

        self.channels = in_dim
        self.feature1 = nn.Linear(in_dim, 6)
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
        # x: [Batch, Input_length, node]
        x = x.permute(0, 2, 1)  # x: [Batch, node, Input_length]
        x = torch.unsqueeze(x, dim=1) # x: [Batch, in_dim, node, Input_length]
        seasonal_init, trend_init = self.decompsition(x)  # x: [Batch, in_dim, node, Input_length]
        if self.individual:
            seasonal_output = torch.zeros(
                [seasonal_init.size(0), seasonal_init.size(1), seasonal_init.size(2), self.Horizon],
                dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), trend_init.size(2), self.Horizon],
                                       dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :, :])
                trend_output[:, i, :, :] = self.Linear_Trend[i](trend_init[:, i, :, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output  # x: [Batch, in_dim, node, Output_length]
        x = x.permute(0, 2, 1, 3)  # x: [Batch, node, in_dim, Output_length]
        x = x.permute(0, 1, 3, 2)  # x: [Batch, node, Output_length, in_dim]
        x = self.feature1(x)
        x = self.feature2(x)  # to [Batch, node, Output_length, Channel]
        x = x[..., -1]  # to [Batch, node, Output_length]
        x = x.permute(0, 2, 1)  # to [Batch, Output_length, out_node]
        return x[...,  -self.out_nodes:]    # to [Batch, Output_length,  out_node]