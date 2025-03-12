import torch
from torch import nn
import torch.nn.functional as F


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncvl,vw->ncwl', (x, A))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out, bias=True):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=bias)

    def forward(self, x):
        return self.mlp(x)


class layer_block(nn.Module):
    def __init__(self, c_in, c_out, k_size):
        super(layer_block, self).__init__()
        self.conv_output = nn.Conv2d(c_in, c_out, kernel_size=(1, 1), stride=(1, 2))

        self.conv_output1 = nn.Conv2d(c_in, c_out, kernel_size=(1, k_size), stride=(1, 1),
                                      padding=(0, int((k_size - 1) / 2)))
        self.output = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))

        self.conv_output1 = nn.Conv2d(c_in, c_out, kernel_size=(1, k_size), stride=(1, 1))
        self.output = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2))
        self.relu = nn.ReLU()

    def forward(self, input):
        conv_output = self.conv_output(input)  # shape (B, D, N, T)

        conv_output1 = self.conv_output1(input)

        output = self.output(conv_output1)

        return self.relu(output + conv_output[..., -output.shape[3]:])

        # return self.relu( conv_output )


class graph_constructor(nn.Module):
    def __init__(self, nnodes, k, dim, layer_num, device, alpha=3):
        super(graph_constructor, self).__init__()
        self.nnodes = nnodes
        self.layers = layer_num

        self.emb1 = nn.Embedding(nnodes, dim)
        self.emb2 = nn.Embedding(nnodes, dim)

        self.lin1 = nn.ModuleList()
        self.lin2 = nn.ModuleList()
        for i in range(layer_num):
            self.lin1.append(nn.Linear(dim, dim))
            self.lin2.append(nn.Linear(dim, dim))

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha

    def forward(self, idx, scale_idx, scale_set):

        nodevec1 = self.emb1(idx)
        nodevec2 = self.emb2(idx)

        adj_set = []

        for i in range(self.layers):
            nodevec1 = torch.tanh(self.alpha * self.lin1[i](nodevec1 * scale_set[i]))
            nodevec2 = torch.tanh(self.alpha * self.lin2[i](nodevec2 * scale_set[i]))
            a = torch.mm(nodevec1, nodevec2.transpose(1, 0)) - torch.mm(nodevec2, nodevec1.transpose(1, 0))
            adj0 = F.relu(torch.tanh(self.alpha * a))

            mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
            mask.fill_(float('0'))
            s1, t1 = adj0.topk(self.k, 1)
            mask.scatter_(1, t1, s1.fill_(1))
            # print(mask)
            adj = adj0 * mask
            adj_set.append(adj)

        return adj_set


class multi_scale_block(nn.Module):
    def __init__(self, c_in, c_out, num_nodes, seq_length, layer_num, kernel_set, layer_norm_affline=True):
        super(multi_scale_block, self).__init__()

        self.seq_length = seq_length
        self.layer_num = layer_num
        self.norm = nn.ModuleList()
        self.scale = nn.ModuleList()

        for i in range(self.layer_num):
            self.norm.append(nn.BatchNorm2d(c_out, affine=False))
        #     # self.norm.append(LayerNorm((c_out, num_nodes, int(self.seq_length/2**i)),elementwise_affine=layer_norm_affline))
        #     self.norm.append(LayerNorm((c_out, num_nodes, length_set[i]),elementwise_affine=layer_norm_affline))

        self.start_conv = nn.Conv2d(c_in, c_out, kernel_size=(1, 1), stride=(1, 1))

        self.scale.append(nn.Conv2d(c_out, c_out, kernel_size=(1, kernel_set[0]), stride=(1, 1)))

        for i in range(1, self.layer_num):
            self.scale.append(layer_block(c_out, c_out, kernel_set[i]))

    def forward(self, input, idx):  # input shape: B D N T

        self.idx = idx

        scale = []
        scale_temp = input

        scale_temp = self.start_conv(scale_temp)
        # scale.append(scale_temp)
        for i in range(self.layer_num):
            scale_temp = self.scale[i](scale_temp)
            # scale_temp = self.norm[i](scale_temp)
            # scale_temp = self.norm[i](scale_temp, self.idx)

            # scale.append(scale_temp[...,-self.k:])
            scale.append(scale_temp)

        return scale


class mixprop(nn.Module):
    def __init__(self, c_in, c_out, gdep, dropout, alpha):
        super(mixprop, self).__init__()
        self.nconv = nconv()
        self.mlp = linear((gdep + 1) * c_in, c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self, x, adj):
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x
        out = [h]
        a = adj / d.view(-1, 1)
        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, a)
            out.append(h)

        ho = torch.cat(out, dim=1)

        ho = self.mlp(ho)

        return ho


class gated_fusion(nn.Module):
    def __init__(self, skip_channels, layer_num, ratio=1):
        super(gated_fusion, self).__init__()
        # self.reduce = torch.mean(x,dim=2,keepdim=True)
        self.dense1 = nn.Linear(in_features=skip_channels * (layer_num + 1), out_features=(layer_num + 1) * ratio,
                                bias=False)

        self.dense2 = nn.Linear(in_features=(layer_num + 1) * ratio, out_features=(layer_num + 1), bias=False)

    def forward(self, input1, input2):
        se = torch.mean(input1, dim=2, keepdim=False)
        se = torch.squeeze(se)

        se = F.relu(self.dense1(se))
        se = F.sigmoid(self.dense2(se))

        se = torch.unsqueeze(se, -1)
        se = torch.unsqueeze(se, -1)
        se = torch.unsqueeze(se, -1)

        x = torch.mul(input2, se)
        x = torch.mean(x, dim=1, keepdim=False)
        return x


class MAGNN(nn.Module):
    def __init__(self, config, adj_mx, node_dim=40,
                 propalpha=0.05, single_step=False):
        super(MAGNN, self).__init__()
        gcn_depth = config['gcn_k']
        num_nodes = config["capacity"]
        out_nodes = config["out_capacity"]
        device = config['device']
        dropout = config.get('dropout', 0.1)
        subgraph_size = config.get('subgraph_size', 20)
        conv_channels = config.get('conv_channels', 32)
        scale_channels = config.get('conv_channels', 32) // 2
        gnn_channels = config.get('gcn_channels', 32)
        in_dim = config["var_len"]
        end_channels = 16
        out_dim = config['out_dim']
        # layers = config.get('layers', 2)
        layers = 2
        seq_length = config["input_len"]
        out_len = config["output_len"]
        self.output_window = out_len
        self.num_nodes = num_nodes
        self.out_node = out_nodes
        self.dropout = dropout

        self.device = device
        self.single_step = single_step
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()

        self.scale_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.norm = nn.ModuleList()

        self.seq_length = seq_length
        self.layer_num = layers

        self.gc = graph_constructor(num_nodes, subgraph_size, node_dim, self.layer_num, device)

        if self.single_step:
            self.kernel_set = [7, 6, 3, 2]
        else:
            # self.kernel_set = [3, 7, 11, 15]   # for test1
            self.kernel_set = [3, 5, 7, 9]   # for test2

        self.scale_id = torch.autograd.Variable(torch.randn(self.layer_num, device=self.device), requires_grad=True)
        # self.scale_id = torch.arange(self.layer_num).to(device)
        self.lin1 = nn.Linear(self.layer_num, self.layer_num)

        self.idx = torch.arange(self.num_nodes).to(device)
        self.scale_idx = torch.arange(self.num_nodes).to(device)

        self.scale0 = nn.Conv2d(in_channels=in_dim, out_channels=scale_channels, kernel_size=(1, self.seq_length),
                                bias=True)

        self.multi_scale_block = multi_scale_block(in_dim, conv_channels, self.num_nodes, self.seq_length,
                                                   self.layer_num, self.kernel_set)
        # self.agcrn = nn.ModuleList()

        # length_set = []
        # length_set.append(self.kernel_set[0])
        # for i in range(1, self.layer_num):
        #     length_set.append(self.kernel_set[i])

        for i in range(self.layer_num):
            """
            RNN based model
            """
            # self.agcrn.append(AGCRN(num_nodes=self.num_nodes, input_dim=conv_channels, hidden_dim=scale_channels, num_layers=1) )

            self.gconv1.append(mixprop(conv_channels, gnn_channels, gcn_depth, dropout, propalpha))
            self.gconv2.append(mixprop(conv_channels, gnn_channels, gcn_depth, dropout, propalpha))

            self.scale_convs.append(nn.Conv2d(in_channels=gnn_channels,
                                              out_channels=scale_channels,
                                              kernel_size=(1, 3)))

        # self.gated_fusion = gated_fusion(scale_channels, self.layer_num)
        self.end_conv_1 = nn.Conv2d(in_channels=scale_channels,
                                    out_channels=end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1, 1),
                                    bias=True)

        if seq_length == 30:  # 临时，只是适用于当前参数，
            # in_feature = 35  # for test1
            in_feature = 36    # for test2
        if seq_length == 60:
            # in_feature = 80  # for test1
            in_feature = 81   # for test2
        if seq_length == 90:
            # in_feature = 125  # for test1
            in_feature = 126   # for test2
        self.out_pool = nn.Linear(in_feature, out_len)  # 临时，只是适用于当前参数，
        # 多尺度计算过程太复杂，没法预先算出输出长度

    def forward(self, input, idx=None):  # (batch_size, input_window, num_nodes)
        input = input.permute((0, 2, 1))  # (batch_size, num_nodes, input_window)
        input = torch.unsqueeze(input, 1)  # (batch_size, 1, num_nodes, input_window)
        seq_len = input.size(3)

        assert seq_len == self.seq_length, 'input sequence length not equal to preset sequence length'

        scale = self.multi_scale_block(input, self.idx)

        # self.scale_weight = self.lin1(self.scale_id)

        self.scale_set = [1, 0.8, 0.6, 0.5]

        adj_matrix = self.gc(self.idx, self.scale_idx, self.scale_set)

        outputs = self.scale0(F.dropout(input, self.dropout, training=self.training))

        out = []
        out.append(outputs)

        for i in range(self.layer_num):
            output = self.gconv1[i](scale[i], adj_matrix[i]) + self.gconv2[i](scale[i], adj_matrix[i].transpose(1, 0))

            scale_specific_output = self.scale_convs[i](output)

            out.append(scale_specific_output)

        outputs = torch.cat(out, dim=-1)

        # x = F.relu(outputs)
        x = self.end_conv_1(outputs)
        x = self.end_conv_2(x)
        x = self.out_pool(x)
        x = x[:, -1, :, :]
        return x.permute(0, 2, 1) # to [Batch, Output_length,  out_node]
