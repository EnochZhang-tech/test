from __future__ import division
import torch
import torch.nn as nn
from torch.nn import init
import numbers
import torch.nn.functional as F
from logging import getLogger
import numpy as np
from compare_model.models import DLinear_model


class NConv(nn.Module):
    def __init__(self):
        super(NConv, self).__init__()

    def forward(self, x, adj):
        """
        A * X

        Args:
            x(torch.tensor):  (B, input_channels, N, T)
            adj(torch.tensor):  N * N

        Returns:
            torch.tensor: (B, input_channels, N, T)
        """
        x = torch.einsum('ncwl,vw->ncvl', (x, adj))
        return x.contiguous()


class DyNconv(nn.Module):
    def __init__(self):
        super(DyNconv, self).__init__()

    def forward(self, x, adj):
        x = torch.einsum('ncvl,nvwl->ncwl', (x, adj))
        return x.contiguous()


class Linear(nn.Module):
    def __init__(self, c_in, c_out, bias=True):
        super(Linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=bias)

    def forward(self, x):
        return self.mlp(x)


class Prop(nn.Module):
    def __init__(self, c_in, c_out, gdep, dropout, alpha):
        super(Prop, self).__init__()
        self.nconv = NConv()
        self.mlp = Linear(c_in, c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self, x, adj):
        """

        Args:
            x(torch.tensor):  (B, c_in, N, T)
            adj(torch.tensor):  N * N

        Returns:
            torch.tensor: (B, c_out, N, T)
        """
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x
        dv = d
        a = adj / dv.view(-1, 1)
        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, a)
        ho = self.mlp(h)
        return ho


class MixProp(nn.Module):
    def __init__(self, c_in, c_out, gdep, dropout, alpha):
        """
        MixProp GCN

        Args:
            c_in: input
            c_out: output
            gdep: GCN layers
            dropout: dropout
            alpha: beta in paper
        """
        super(MixProp, self).__init__()
        self.nconv = NConv()
        self.mlp = Linear((gdep + 1) * c_in, c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self, x, adj):
        """
        MixProp GCN

        Args:
            x(torch.tensor):  (B, c_in, N, T)
            adj(torch.tensor):  N * N

        Returns:
            torch.tensor: (B, c_out, N, T)
        """
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x
        out = [h]  # h(0) = h_in = x
        a = adj / d.view(-1, 1)  # A' = A * D^-1
        for i in range(self.gdep):
            # h(k) = alpha * h_in + (1 - alpha) * A' * H(k-1)
            # h: shape = (B, c_in, N, T)
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, a)
            out.append(h)
        # ho: (B, c_in * (gdep + 1), N, T)
        ho = torch.cat(out, dim=1)
        ho = self.mlp(ho)
        return ho  # (B, c_out, N, T)


class DyMixprop(nn.Module):
    def __init__(self, c_in, c_out, gdep, dropout, alpha):
        super(DyMixprop, self).__init__()
        self.nconv = DyNconv()
        self.mlp1 = Linear((gdep + 1) * c_in, c_out)
        self.mlp2 = Linear((gdep + 1) * c_in, c_out)

        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha
        self.lin1 = Linear(c_in, c_in)
        self.lin2 = Linear(c_in, c_in)

    def forward(self, x):
        x1 = torch.tanh(self.lin1(x))
        x2 = torch.tanh(self.lin2(x))
        adj = self.nconv(x1.transpose(2, 1), x2)
        adj0 = torch.softmax(adj, dim=2)
        adj1 = torch.softmax(adj.transpose(2, 1), dim=2)

        h = x
        out = [h]
        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, adj0)
            out.append(h)
        ho = torch.cat(out, dim=1)
        ho1 = self.mlp1(ho)

        h = x
        out = [h]
        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, adj1)
            out.append(h)
        ho = torch.cat(out, dim=1)
        ho2 = self.mlp2(ho)
        return ho1 + ho2


class GraphConstructor(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(GraphConstructor, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim, dim)
            self.lin2 = nn.Linear(dim, dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx, :]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1, 0)) - torch.mm(nodevec2, nodevec1.transpose(1, 0))
        adj = F.relu(torch.tanh(self.alpha * a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float('0'))
        s1, t1 = adj.topk(self.k, 1)
        mask.scatter_(1, t1, s1.fill_(1))
        adj = adj * mask
        return adj

    def fulla(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx, :]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1, 0)) - torch.mm(nodevec2, nodevec1.transpose(1, 0))
        adj = F.relu(torch.tanh(self.alpha * a))
        return adj


class GraphGlobal(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(GraphGlobal, self).__init__()
        self.nnodes = nnodes
        self.A = nn.Parameter(torch.randn(nnodes, nnodes).to(device), requires_grad=True).to(device)

    def forward(self, idx):
        return F.relu(self.A)


class GraphUndirected(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(GraphUndirected, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim, dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb1(idx)
        else:
            nodevec1 = self.static_feat[idx, :]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin1(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1, 0))
        adj = F.relu(torch.tanh(self.alpha * a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float('0'))
        s1, t1 = adj.topk(self.k, 1)
        mask.scatter_(1, t1, s1.fill_(1))
        adj = adj * mask
        return adj


class GraphDirected(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(GraphDirected, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim, dim)
            self.lin2 = nn.Linear(dim, dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx, :]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1, 0))
        adj = F.relu(torch.tanh(self.alpha * a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float('0'))
        s1, t1 = adj.topk(self.k, 1)
        mask.scatter_(1, t1, s1.fill_(1))
        adj = adj * mask
        return adj


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
    def __init__(self, input_len, output_len, in_dim, ks=None, individual=None):
        super(DLinear, self).__init__()
        self.Lag = input_len
        self.Horizon = output_len
        self.individual = individual

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
        # x: [Batch, in_dim, node, Input_length]
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
        return x[..., -1]


class MixProp_Dlinear(nn.Module):
    def __init__(self, config, adj_mx=None):
        super().__init__()
        self.adj_mx = np.array(adj_mx)
        self.feature_dim = config["var_len"]
        self.input_window = config["input_len"]
        self.output_window = config["output_len"]
        self.num_nodes = config["capacity"]
        self.out_nodes = config['out_capacity']
        self.data_diff = config["data_diff"]
        self.dilation_0 = config.get('dilation_0', 1)
        self.output_dim = 1
        self.device = config['device']

        self.gcn_true = config.get('gcn_true', True)
        self.add_apt = config.get('add_apt', False)
        self.gcn_depth = config['gcn_k']
        self.dropout = config.get('dropout', 0.1)
        self.subgraph_size = config.get('subgraph_size', 20)
        self.node_dim = config.get('node_dim', 40)
        self.conv_channels = config.get('conv_channels', 32)
        self.residual_channels = config.get('residual_channels', 32)
        self.layers = config.get('layers', 3)
        self.propalpha = config.get('propalpha', 0.05)
        self.tanhalpha = config.get('tanhalpha', 3)
        self.layer_norm_affline = config.get('layer_norm_affline', True)
        self.static_feat = None
        self.idx = torch.arange(self.num_nodes).to(self.device)

        self._logger = getLogger()

        self.predefined_A = torch.tensor(self.adj_mx.astype(np.float32))
        self.predefined_A = self.predefined_A.to(self.device)

        self.start_conv = nn.Conv2d(in_channels=self.feature_dim,
                                    out_channels=self.residual_channels,
                                    kernel_size=(1, 1))
        self.gconv = MixProp(self.residual_channels, self.conv_channels,
                             self.gcn_depth, self.dropout, self.propalpha)
        self.dlinear = DLinear(self.input_window, self.output_window, in_dim=self.conv_channels, ks=7)

        if self.add_apt:
            self.gc = GraphConstructor(self.num_nodes, self.subgraph_size, self.node_dim,
                                       self.device, alpha=self.tanhalpha, static_feat=self.static_feat)


    def forward(self, batch_x):  # (bs, T, Node)
        inputs = batch_x.permute((0, 2, 1))  # (bs, Node, T)
        inputs = torch.unsqueeze(inputs, dim=-1)  # (batch_size, num_nodes, input_window, feature_dim)
        # inputs = batch_x.permute((0, 2, 1, 3))
        bz, id_len, input_len, var_len = inputs.shape

        # inputs = inputs[:, :, :, 2:]

        if self.data_diff != 0:
            # add Data Differential Features
            diff_data = []
            inputs_diff = inputs
            for d in range(self.data_diff):
                inputs_diff = inputs_diff[:, :, 1:, :] - inputs_diff[:, :, :-1, :]
                inputs_diff = torch.cat((torch.zeros(bz, id_len, 1, 1).to(inputs.device), inputs_diff), 2)
                diff_data.append(inputs_diff)
            inputs_diff = torch.cat(diff_data, dim=3)
            inputs = torch.cat((inputs, inputs_diff), 3)

        inputs = inputs.permute((0, 3, 1, 2))  # (batch_size, feature_dim, num_nodes, input_window)

        assert inputs.size(3) == self.input_window, 'input sequence length not equal to preset sequence length'
        x = self.start_conv(inputs)
        if self.gcn_true:
            if self.add_apt:
                adp = self.gc(self.idx)
            else:
                adp = self.predefined_A

        x = self.gconv(x, adp)
        x = self.dlinear(x)

        return x.permute(0, 2, 1)[..., -self.out_nodes:]  # (B, T, N)
