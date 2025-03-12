import torch as tr
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import OrderedDict


class Feature_extractor_1DCNN_RUL(nn.Module):
    def __init__(self, input_channels, num_hidden, out_dim, kernel_size=8, stride=1, dropout=0):
        super(Feature_extractor_1DCNN_RUL, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(input_channels, num_hidden, kernel_size=kernel_size,
                      stride=stride, bias=False, padding=(kernel_size // 2)),
            nn.BatchNorm1d(num_hidden),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(num_hidden, out_dim, kernel_size=kernel_size, stride=1, bias=False, padding=1),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

    def forward(self, x_in):
        # print('input size is {}'.format(x_in.size()))
        ### input dim is (bs, tlen, feature_dim)
        x = tr.transpose(x_in, -1, -2)

        x = self.conv_block1(x)
        x = self.conv_block2(x)

        return x


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = tr.zeros(max_len, d_model).cuda()
        position = tr.arange(0, max_len).unsqueeze(1)
        div_term = tr.exp(tr.arange(0, d_model, 2) *
                          -(math.log(100.0) / d_model))
        pe[:, 0::2] = tr.sin(position * div_term)
        pe[:, 1::2] = tr.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x = x + torch.Tensor(self.pe[:, :x.size(1)],
        #                  requires_grad=False)
        # print(self.pe[0, :x.size(1),2:5])
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
        # return x


class Dot_Graph_Construction_weights(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.mapping = nn.Linear(input_dim, input_dim)

    def forward(self, node_features):
        node_features = self.mapping(node_features)
        # node_features = F.leaky_relu(node_features)
        bs, N, dimen = node_features.size()

        node_features_1 = tr.transpose(node_features, 1, 2)

        Adj = tr.bmm(node_features, node_features_1)

        eyes_like = tr.eye(N).repeat(bs, 1, 1).cuda()
        eyes_like_inf = eyes_like * 1e8
        Adj = F.leaky_relu(Adj - eyes_like_inf)
        Adj = F.softmax(Adj, dim=-1)
        # print(Adj[0])
        Adj = Adj + eyes_like
        # print(Adj[0])
        # if prior:

        return Adj


class MPNN_mk_v2(nn.Module):
    def __init__(self, input_dimension, outpuut_dinmension, k):
        ### In GCN, k means the size of receptive field. Different receptive fields can be concatnated or summed
        ### k=1 means the traditional GCN
        super(MPNN_mk_v2, self).__init__()
        self.way_multi_field = 'sum'  ## two choices 'cat' (concatnate) or 'sum' (sum up)
        self.k = k
        theta = []
        for kk in range(self.k):
            theta.append(nn.Linear(input_dimension, outpuut_dinmension))
        self.theta = nn.ModuleList(theta)
        self.bn1 = nn.BatchNorm1d(outpuut_dinmension)

    def forward(self, X, A):
        ## size of X is (bs, N, A)
        ## size of A is (bs, N, N)
        GCN_output_ = []
        for kk in range(self.k):
            if kk == 0:
                A_ = A
            else:
                A_ = tr.bmm(A_, A)
            out_k = self.theta[kk](tr.bmm(A_, X))
            GCN_output_.append(out_k)

        if self.way_multi_field == 'cat':
            GCN_output_ = tr.cat(GCN_output_, -1)

        elif self.way_multi_field == 'sum':
            GCN_output_ = sum(GCN_output_)

        GCN_output_ = tr.transpose(GCN_output_, -1, -2)
        GCN_output_ = self.bn1(GCN_output_)
        GCN_output_ = tr.transpose(GCN_output_, -1, -2)

        return F.leaky_relu(GCN_output_)


def Mask_Matrix(num_node, time_length, decay_rate):
    Adj = tr.ones(num_node * time_length, num_node * time_length).cuda()
    for i in range(time_length):
        v = 0
        for r_i in range(i, time_length):
            idx_s_row = i * num_node
            idx_e_row = (i + 1) * num_node
            idx_s_col = (r_i) * num_node
            idx_e_col = (r_i + 1) * num_node
            Adj[idx_s_row:idx_e_row, idx_s_col:idx_e_col] = Adj[idx_s_row:idx_e_row, idx_s_col:idx_e_col] * (
                        decay_rate ** (v))
            v = v + 1
        v = 0
        for r_i in range(i + 1):
            idx_s_row = i * num_node
            idx_e_row = (i + 1) * num_node
            idx_s_col = (i - r_i) * num_node
            idx_e_col = (i - r_i + 1) * num_node
            Adj[idx_s_row:idx_e_row, idx_s_col:idx_e_col] = Adj[idx_s_row:idx_e_row, idx_s_col:idx_e_col] * (
                        decay_rate ** (v))
            v = v + 1

    return Adj


def Conv_GraphST(input, time_window_size, stride):
    ## input size is (bs, time_length, num_sensors, feature_dim)
    ## output size is (bs, num_windows, num_sensors, time_window_size, feature_dim)
    bs, time_length, num_sensors, feature_dim = input.size()
    x_ = tr.transpose(input, 1, 3)

    y_ = F.unfold(x_, (num_sensors, time_window_size), stride=stride)

    y_ = tr.reshape(y_, [bs, feature_dim, num_sensors, time_window_size, -1])
    y_ = tr.transpose(y_, 1, -1)

    return y_


class GraphConvpoolMPNN_block_v6(nn.Module):
    def __init__(self, input_dim, output_dim, num_sensors, time_window_size, stride, decay, pool_choice):
        super(GraphConvpoolMPNN_block_v6, self).__init__()
        self.time_window_size = time_window_size
        self.stride = stride
        self.output_dim = output_dim

        self.graph_construction = Dot_Graph_Construction_weights(input_dim)
        self.BN = nn.BatchNorm1d(input_dim)

        self.MPNN = MPNN_mk_v2(input_dim, output_dim, k=1)

        self.pre_relation = Mask_Matrix(num_sensors, time_window_size, decay)

        self.pool_choice = pool_choice

    def forward(self, input):
        ## input size (bs, time_length, num_nodes, input_dim)
        ## output size (bs, output_node_t, output_node_s, output_dim)

        input_con = Conv_GraphST(input, self.time_window_size, self.stride)
        ## input_con size (bs, num_windows, num_sensors, time_window_size, feature_dim)
        bs, num_windows, num_sensors, time_window_size, feature_dim = input_con.size()
        input_con_ = tr.transpose(input_con, 2, 3)
        input_con_ = tr.reshape(input_con_, [bs * num_windows, time_window_size * num_sensors, feature_dim])

        A_input = self.graph_construction(input_con_)
        # print(A_input.size())
        # print(self.pre_relation.size())
        A_input = A_input * self.pre_relation

        input_con_ = tr.transpose(input_con_, -1, -2)
        input_con_ = self.BN(input_con_)
        input_con_ = tr.transpose(input_con_, -1, -2)
        X_output = self.MPNN(input_con_, A_input)

        X_output = tr.reshape(X_output, [bs, num_windows, time_window_size, num_sensors, self.output_dim])
        # print(X_output.size())

        if self.pool_choice == 'mean':
            X_output = tr.mean(X_output, 2)
        elif self.pool_choice == 'max':

            X_output, ind = tr.max(X_output, 2)
        else:
            print('input choice for pooling cannot be read')
        # X_output = tr.reshape(X_output, [bs, num_windows*time_window_size,num_sensors, self.output_dim])
        # print(X_output.size())

        return X_output


class FCSTGNN(nn.Module):
    def __init__(self, config, adj=None, n_class=1):
        super(FCSTGNN, self).__init__()
        # graph_construction_type = args.graph_construction_type
        indim_fea = config.var_len
        Conv_out = config.conv_out
        lstmhidden_dim = config.lstmhidden_dim
        lstmout_dim = config.lstmout_dim
        conv_kernel = config.conv_kernel
        hidden_dim = config.hid_dim
        num_node = config.capacity
        num_windows = config.num_windows
        moving_window = config.moving_window
        stride = config.stride
        decay = config.decay
        pooling_choice = config.pool_choice

        self.nonlin_map = Feature_extractor_1DCNN_RUL(indim_fea, lstmhidden_dim, lstmout_dim, kernel_size=conv_kernel)
        self.nonlin_map2 = nn.Sequential(
            nn.Linear(lstmout_dim * Conv_out, 2 * hidden_dim),
            nn.BatchNorm1d(2 * hidden_dim)
        )

        self.positional_encoding = PositionalEncoding(2 * hidden_dim, 0.1, max_len=5000)

        self.MPNN1 = GraphConvpoolMPNN_block_v6(2 * hidden_dim, hidden_dim, num_node,
                                                time_window_size=moving_window[0], stride=stride[0], decay=decay,
                                                pool_choice=pooling_choice)
        self.MPNN2 = GraphConvpoolMPNN_block_v6(2 * hidden_dim, hidden_dim, num_node,
                                                time_window_size=moving_window[1], stride=stride[1], decay=decay,
                                                pool_choice=pooling_choice)

        self.fc = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(672, 300)),
            ('relu1', nn.ReLU(inplace=True)),
            ('fc2', nn.Linear(300, 150)),
            ('relu2', nn.ReLU(inplace=True)),
            ('fc3', nn.Linear(150, 60)),
            ('relu3', nn.ReLU(inplace=True)),
            ('fc4', nn.Linear(60, config.output_len)),

        ]))

    def forward(self, X):
        # X: [bs, seq_len, node]
        # print(X.size())
        X = tr.unsqueeze(X, dim=-1)
        bs, tlen, num_node, dimension = X.size()  ### [bs, seq_len, node, dim]

        ### Graph Generation
        A_input = tr.reshape(X, [bs * tlen * num_node, dimension, 1])  ### [bs*seq_len*node, dim, 1]
        A_input_ = self.nonlin_map(A_input)  ### [bs*seq_len*node, lstmout_dim, 3]
        A_input_ = tr.reshape(A_input_, [bs * tlen * num_node, -1])  ### [bs*seq_len*node, lstmout_dim*3]
        A_input_ = self.nonlin_map2(A_input_)  ### [bs*seq_len*node, 2 * hidden_dim]
        A_input_ = tr.reshape(A_input_, [bs, tlen, num_node, -1])  ### [bs, seq_len, node, 2 * hidden_dim]

        # print('A_input size is ', A_input_.size())

        ## positional encoding before mapping starting
        X_ = tr.reshape(A_input_, [bs, tlen, num_node, -1])  ### [bs, seq_len, node, 2 * hidden_dim]
        X_ = tr.transpose(X_, 1, 2)  ### [bs, node, seq_len, 2 * hidden_dim]
        X_ = tr.reshape(X_, [bs * num_node, tlen, -1])  ### [bs*node, seq_len, 2 * hidden_dim]
        X_ = self.positional_encoding(X_)  ### [bs*node, seq_len, 2 * hidden_dim]
        X_ = tr.reshape(X_, [bs, num_node, tlen, -1])  ### [bs, node, seq_len, 2 * hidden_dim]
        X_ = tr.transpose(X_, 1, 2)  ### [bs, seq_len, node, 2 * hidden_dim]
        A_input_ = X_

        ## positional encoding before mapping ending

        MPNN_output1 = self.MPNN1(A_input_)
        MPNN_output2 = self.MPNN2(A_input_)

        features1 = tr.permute(MPNN_output1, dims=(0, 2, 1, 3))
        features2 = tr.permute(MPNN_output2, dims=(0, 2, 1, 3))
        features1 = tr.reshape(features1, shape=(bs, features1.shape[1], -1))
        features2 = tr.reshape(features2, shape=(bs, features2.shape[1], -1))

        features = tr.cat([features1, features2], -1)

        features = self.fc(features)
        features = tr.permute(features, dims=(0, 2, 1))
        return features
