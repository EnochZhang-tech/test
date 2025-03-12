from __future__ import division
import torch
import torch.nn as nn
from torch.nn import init
import numbers
import torch.nn.functional as F
from logging import getLogger
import numpy as np


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


class Linear(nn.Module):
    def __init__(self, c_in, c_out, bias=True):
        super(Linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=bias)

    def forward(self, x):
        return self.mlp(x)


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


class DilatedInception(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super(DilatedInception, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [2, 3, 6, 7]
        cout = int(cout / len(self.kernel_set))
        for kern in self.kernel_set:
            self.tconv.append(nn.Conv2d(cin, cout, (1, kern), dilation=(1, dilation_factor)))

    def forward(self, input):
        """

        Args:
            inputs: (B, C_in, N, T)

        Returns:
            torch.tensor: (B, C_out, N, T)

        """
        x = []
        for i in range(len(self.kernel_set)):
            x.append(self.tconv[i](input))
        for i in range(len(self.kernel_set)):
            x[i] = x[i][..., -x[-1].size(3):]
        x = torch.cat(x, dim=1)
        return x


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


class LayerNorm(nn.Module):
    __constants__ = ['normalized_shape', 'weight', 'bias', 'eps', 'elementwise_affine']

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(*normalized_shape))
            self.bias = nn.Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, inputs, idx):
        if self.elementwise_affine:
            return F.layer_norm(inputs, tuple(inputs.shape[1:]),
                                self.weight[:, idx, :], self.bias[:, idx, :], self.eps)
        else:
            return F.layer_norm(inputs, tuple(inputs.shape[1:]),
                                self.weight, self.bias, self.eps)

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
               'elementwise_affine={elementwise_affine}'.format(**self.__dict__)


class Ablation(nn.Module):
    def __init__(self, config, adj_mx=None):
        super().__init__()
        self.gcn_true = config.gcn
        self.tcn = config.tcn
        self.hd = config.hd

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

        self.add_apt = config.get('add_apt', False)
        self.gcn_depth = config['gcn_k']
        self.dropout = config.get('dropout', 0.1)
        self.subgraph_size = config.get('subgraph_size', 20)
        self.node_dim = config.get('node_dim', 40)
        self.dilation_exponential = config.get('dilation_exponential', 2)
        self.conv_channels = config.get('conv_channels', 32)
        self.residual_channels = config.get('residual_channels', 32)
        self.skip_channels = config.get('skip_channels', 64)
        self.end_channels = config.get('end_channels', 128)
        self.layers = config.get('layers', 3)
        self.propalpha = config.get('propalpha', 0.05)
        self.tanhalpha = config.get('tanhalpha', 3)
        self.layer_norm_affline = config.get('layer_norm_affline', True)

        self.static_feat = None
        self.idx = torch.arange(self.num_nodes).to(self.device)

        self._logger = getLogger()

        self.predefined_A = torch.tensor(self.adj_mx.astype(np.float32))
        self.predefined_A = self.predefined_A.to(self.device)
        self.static_feat = None
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.gconv = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.start_conv = nn.Conv2d(in_channels=self.feature_dim,
                                    out_channels=self.residual_channels,
                                    kernel_size=(1, 1))
        if self.add_apt:
            self.gc = GraphConstructor(self.num_nodes, self.subgraph_size, self.node_dim,
                                       self.device, alpha=self.tanhalpha, static_feat=self.static_feat)

        kernel_size = 7
        if self.dilation_exponential > 1:
            self.receptive_field = int(
                self.output_dim + (kernel_size - 1) * (self.dilation_exponential ** self.layers - 1)
                / (self.dilation_exponential - 1))
        else:
            self.receptive_field = self.layers * (kernel_size - 1) + self.output_dim

        for i in range(1):
            if self.dilation_exponential > 1:
                rf_size_i = int(1 + i * (kernel_size - 1) * (self.dilation_exponential ** self.layers - 1)
                                / (self.dilation_exponential - 1))
            else:
                rf_size_i = i * self.layers * (kernel_size - 1) + 1
            new_dilation = self.dilation_0
            for j in range(1, self.layers + 1):
                if self.dilation_exponential > 1:
                    rf_size_j = int(rf_size_i + (kernel_size - 1) * (self.dilation_exponential ** j - 1)
                                    / (self.dilation_exponential - 1))
                else:
                    rf_size_j = rf_size_i + j * (kernel_size - 1)

                self.filter_convs.append(DilatedInception(self.residual_channels,
                                                          self.conv_channels, dilation_factor=new_dilation))
                self.gate_convs.append(DilatedInception(self.residual_channels,
                                                        self.conv_channels, dilation_factor=new_dilation))
                self.residual_convs.append(nn.Conv2d(in_channels=self.conv_channels,
                                                     out_channels=self.residual_channels, kernel_size=(1, 1)))

                if self.gcn_true:
                    self.gconv.append(MixProp(self.conv_channels, self.residual_channels,
                                              self.gcn_depth, self.dropout, self.propalpha))

                if self.input_window > self.receptive_field:
                    self.norm.append(LayerNorm((self.residual_channels, self.num_nodes,
                                                self.input_window - rf_size_j + 1 - (self.dilation_0 - 1) * 6),
                                               elementwise_affine=self.layer_norm_affline))
                else:
                    self.norm.append(LayerNorm((self.residual_channels, self.num_nodes,
                                                self.receptive_field - rf_size_j + 1),
                                               elementwise_affine=self.layer_norm_affline))

                new_dilation *= self.dilation_exponential

        if self.input_window > self.receptive_field:
            self.end_conv = nn.Conv2d(in_channels=self.residual_channels,
                                      out_channels=self.skip_channels,
                                      kernel_size=(
                                          1, self.input_window - self.receptive_field + 1 - (self.dilation_0 - 1) * 6),
                                      bias=True)
        else:
            self.end_conv = nn.Conv2d(in_channels=self.residual_channels,
                                      out_channels=self.skip_channels, kernel_size=(1, 1), bias=True)
        if self.gcn_true and (not self.tcn):  # 只有GCN， 没有TCN
            self.end_conv = nn.Conv2d(in_channels=self.residual_channels,
                                      out_channels=self.skip_channels, kernel_size=(1, self.input_window), bias=True)

        self.end_conv_1 = nn.Conv2d(in_channels=self.skip_channels,
                                    out_channels=self.end_channels, kernel_size=(1, 1), bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=self.end_channels,
                                    out_channels=self.output_window, kernel_size=(1, 1), bias=True)
        self._logger.info('receptive_field: ' + str(self.receptive_field))

    def forward(self, batch_x):  # (bs, T, Node)
        if not self.hd:
            batch_x = batch_x[..., self.out_nodes:]

        inputs = batch_x.permute((0, 2, 1))  # (bs, Node, T)
        inputs = torch.unsqueeze(inputs, dim=-1)  # (batch_size, num_nodes, input_window, feature_dim=1)
        bz, id_len, input_len, var_len = inputs.shape

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

        inputs = inputs.permute((0, 3, 1, 2))  # (batch_size, var_len, num_nodes, input_window)

        assert inputs.size(3) == self.input_window, 'input sequence length not equal to preset sequence length'

        if self.input_window < self.receptive_field:
            inputs = nn.functional.pad(inputs, (self.receptive_field - self.input_window, 0, 0, 0))

        if self.gcn_true:
            if self.add_apt:
                adp = self.gc(self.idx)
            else:
                adp = self.predefined_A

        x = self.start_conv(inputs)  # (4->32, ks=(1,1))  [bs, 32, node, seq_len=60]
        for i in range(self.layers):
            residual = x
            if self.tcn:
                filters = self.filter_convs[i](x)
                filters = torch.tanh(filters)
                gate = self.gate_convs[i](x)
                gate = torch.sigmoid(gate)
                x = filters * gate  # [bs, 32, node, seq_len=54]
                x = F.dropout(x, self.dropout, training=self.training)

            if self.gcn_true:
                x = self.gconv[i](x, adp)  # [bs, 32, node, seq_len=54]
            else:
                x = self.residual_convs[i](x)  # (32->32, ks=(1,1))

            x = x + residual[:, :, :, -x.size(3):]
            # x = self.norm[i](x, self.idx)
        x = F.relu(self.end_conv(x))  # (32->64, ks=(1,54))
        x = F.relu(self.end_conv_1(x))  # (64->128, ks=(1,1))
        x = self.end_conv_2(x)  # (B,T,N,F)  # (128->pred_len, ks=(1,1))
        return x[..., -self.out_nodes:, 0]  # (B, T, N)
