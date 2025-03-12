import torch
from torch import nn
import numpy as np
from scipy import sparse as sp
import torch.nn.functional as F


class Cheb_conv(nn.Module):
    def __init__(self, K, in_channels, out_channels, device):
        super(Cheb_conv, self).__init__()
        self.K = K
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.DEVICE = device
        self.Theta = nn.ParameterList(
            [nn.Parameter(torch.randn((in_channels, out_channels)).to(self.DEVICE)) for _ in range(K)])

    def forward(self, x, adj):
        '''
        :param x: (bs, c, 36, t)
        :return:
        '''
        # mask = torch.eye(adj.shape[0], device='cuda:0')
        # mask = torch.logical_not(mask).to(torch.float32)
        # adj = mask * adj
        x = torch.permute(x, dims=(0, 2, 1, 3))  # (bs, 36, c, t)
        adj = adj.cpu().detach().numpy()
        L = torch.from_numpy(self.calculate_normalized_laplacian(adj)).to(self.DEVICE)
        cheb_ploynomials = [torch.eye(24).to(self.DEVICE), L]  # adj_mx -> lap
        for k in range(2, self.K):
            cheb_ploynomials.append(torch.matmul(2 * L, cheb_ploynomials[-1]) - cheb_ploynomials[-2])

        batch_size, num_of_vertices, in_channels, num_of_timesteps = x.shape
        outputs = []

        for time_step in range(num_of_timesteps):

            graph_signal = x[:, :, :, time_step]  # (b, N, F_in)

            output = torch.zeros(batch_size, num_of_vertices, self.out_channels).to(self.DEVICE)  # (b, N, F_out)

            for k in range(self.K):
                T_k = cheb_ploynomials[k]  # (N,N)
                # T_k = mask * T_k

                theta_k = self.Theta[k]  # (in_channel, out_channel)

                rhs = graph_signal.permute(0, 2, 1).matmul(T_k).permute(0, 2, 1)

                output = output + rhs.matmul(theta_k)

            outputs.append(output.unsqueeze(-1))

        out = torch.cat(outputs, dim=-1)
        out = torch.permute(out, dims=(0, 2, 1, 3))
        return out

    def calculate_normalized_laplacian(self, adj):  # 该函数每一步弄清楚
        """
        拉普拉斯矩阵的计算
        # L = D^-1/2 A D^-1/2
        # D = diag(A 1)
        :param adj:
        :return:
        """
        # adj[adj < 0] = 0
        adj = sp.coo_matrix(adj)  # 用指定数据生成矩阵
        d = np.array(adj.sum(1))
        d_inv_sqrt = np.power(d, -0.5).flatten()  # 数组元素求n次方
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        normalized_laplacian = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()  # 将稠密矩阵转为稀疏矩阵
        normalized_laplacian = normalized_laplacian.toarray()
        return normalized_laplacian  # 標準化

class TCN(nn.Module):
    def __init__(self, config):#input_channels: int, kernel_size: int, tcn_layers: list, d0=1
        super(TCN, self).__init__()

        self.input_channels=input_channels=config.capacity
        self.output_channels=config.out_capacity
        self.kernel_size=kernel_size=config.ks
        self.d0 = d0=1
        self.tcn_layers = tcn_layers=config.tcn_layers+[self.output_channels]

        out_channels0 = tcn_layers[0]
        stride = 1
        dilation0 = d0
        padding0 = (kernel_size - 1) * dilation0

        self.downsample0 = nn.Conv2d(input_channels, out_channels0, 1) if input_channels != out_channels0 else None
        self.relu = nn.ReLU()

        in_channels1 = tcn_layers[0]
        out_channels1 = tcn_layers[1]
        dilation1 = 2 * d0
        padding1 = (kernel_size - 1) * dilation1

        self.downsample1 = nn.Conv2d(in_channels1, out_channels1, 1) if in_channels1 != out_channels1 else None

        self.conv_block1 = nn.Sequential(
            CausalConv2d(input_channels, out_channels0, kernel_size=(1, kernel_size), stride=stride, padding=padding0,
                         dilation=dilation0),
            # Chomp1d(padding0),
            nn.BatchNorm2d(out_channels0),
            nn.ReLU(),

            CausalConv2d(out_channels0, out_channels0, kernel_size=(1, kernel_size), stride=stride, padding=padding0,
                         dilation=dilation0),
            # Chomp1d(padding0),
            nn.BatchNorm2d(out_channels0),
            nn.ReLU(),
        )

        self.conv_block2 = nn.Sequential(
            CausalConv2d(out_channels0, out_channels1, kernel_size=(1, kernel_size), stride=stride, padding=padding1,
                         dilation=dilation1),
            # Chomp1d(padding1),
            nn.BatchNorm2d(out_channels1),
            nn.ReLU(),

            CausalConv2d(out_channels1, out_channels1, kernel_size=(1, kernel_size), stride=stride, padding=padding1,
                         dilation=dilation1),
            # Chomp1d(padding1),
            nn.BatchNorm2d(out_channels1),
            nn.ReLU(),
        )

        # self.end_conv = nn.Conv1d(out_channels1, out_channels1, kernel_size=input_len - end_stride * (pred_len - 1),
        #                           stride=end_stride)

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        x0 = self.conv_block1(inputs)
        res0 = inputs if self.downsample0 is None else self.downsample0(inputs)
        out_0 = self.relu(x0 + res0)

        x1 = self.conv_block2(out_0)
        res1 = out_0 if self.downsample1 is None else self.downsample1(out_0)
        # out_1 = self.relu(x1 + res1)
        out = x1 + res1
        # out_1 = self.end_conv(out_1)
        return out


class CausalConv2d(nn.Conv2d):
    def __init__(self, in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1):
        super(CausalConv2d, self).__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                           stride=stride, dilation=dilation)
        self.causal_padding = padding

    def forward(self, x):
        """
        对输入进行因果填充并应用卷积。
        """
        # 使用 F.pad 进行因果填充，填充在序列的开始处
        x = F.pad(x, (self.causal_padding, 0))

        # 调用父类的 forward 方法来应用卷积
        return super(CausalConv2d, self).forward(x)


class GCN_TCN(nn.Module):
    def __init__(self, config, adj_mx=None):
        super().__init__()
        self.num_node = config.capacity
        self.device = config.device
        self.add_apt = config.add_apt

        if config.add_apt:  # 自适应邻接矩阵
            embed_dim = 2
            node_embed_1 = self._init_parameters(
                nn.Parameter(torch.FloatTensor(self.num_node, embed_dim), requires_grad=True))
            node_embed_2 = self._init_parameters(
                nn.Parameter(torch.FloatTensor(embed_dim, self.num_node), requires_grad=True))
            self.predefined_A = node_embed_1 @ node_embed_2
            self.predefined_A = self.predefined_A.to(self.device)
            self.adj_mask = torch.eye(self.num_node, device=config.device)
            self.adj_mask = torch.logical_not(self.adj_mask).to(torch.float32)
        else:
            self.adj_mx = adj_mx
            self.predefined_A = torch.tensor(self.adj_mx.values.astype(np.float32))  # 自定义图
            self.predefined_A = self.predefined_A.to(self.device)

        self.in_channel = config.var_len
        self.input_len = config.input_len
        self.pred_len = config.output_len

        self.gcn_depth = config.gcn_k
        self.dropout = config.dropout
        self.layers = config.layers
        self.kernel_size = config.ks
        self.num_channels = config.channels_list
        self.d0 = config.d0

        self.gc1 = Cheb_conv(self.gcn_depth, in_channels=self.in_channel, out_channels=self.num_channels[0],
                             device=self.device)
        self.tc1 = TCN(input_channels=self.num_channels[0], kernel_size=self.kernel_size, tcn_layers=self.num_channels,
                       d0=self.d0)

        # if self.layers > 1:
        #     self.more_layers = nn.ModuleList()
        #     for l in range(1, self.layers):
        #         self.more_layers.append(TCN(self.num_channels[-1] // 2,
        #                                                 [self.num_channels[-1] // 4, self.num_channels[-1] // 2],
        #                                                 self.kernel_size,
        #                                                 self.dropout))
        #         self.more_layers.append(Cheb_conv(self.gcn_depth, in_channels=self.num_channels[-1] // 2,
        #                                           out_channels=self.num_channels[-1] // 2,
        #                                           device=self.device))

        self.end = nn.Conv2d(self.num_channels[-1], self.num_channels[-1] // 2, kernel_size=(1, 1))
        self.end_1 = nn.Conv2d(self.num_channels[-1] // 2, 1, kernel_size=(1, 2))

        self.relu = nn.ReLU()

    def forward(self, batch_x):  # (32, 10, 24)
        if self.add_apt:
            self.predefined_A = self.adj_mask * self.predefined_A

        batch_x = torch.permute(batch_x, dims=(0, 2, 1))  # (32, 24, 10)
        batch_x = torch.unsqueeze(batch_x, dim=1)  # (32, 1, 24, 10)

        output = self.gc1(batch_x, self.predefined_A)  # (32, 16, 24, 10)
        output = self.tc1(output)  # (32, 32, 24, 10)

        output = self.end(output)  # (bs, c, 24, 10)
        output = self.relu(output)
        output = self.end_1(output)  # (bs, 1, 36, t_len)
        output = torch.squeeze(output, dim= 1)   # (bs, 36, t_len)
        output = output[..., -self.pred_len:]  # (bs, 36, out_len)
        output = output.permute(0, 2, 1)  # (bs, out_len, 36)
        return output

    def _init_parameters(self, parameters):
        if parameters.dim() > 1:
            parameters = nn.init.xavier_uniform_(parameters)
        else:
            parameters = nn.init.uniform_(parameters)
        return parameters
