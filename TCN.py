import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F





class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout):
        self.dilation = dilation
        self.stride = stride
        self.padding = padding
        self.n_outputs = n_outputs

        super(TemporalBlock, self).__init__()
        self.conv1 = (CausalConv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        #经过conv1，输出的size其实是(Batch, input_channel, seq_len + padding)
        #裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.BatchNorm1 = nn.BatchNorm1d(n_outputs)
        # self.relu1 = nn.LeakyReLU(negative_slope=0.5)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
            #最后输出长度的定义
        self.conv2 = (CausalConv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        #裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.BatchNorm2 = nn.BatchNorm1d(n_outputs)
        # self.relu2 = nn.LeakyReLU(negative_slope=0.5)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        #if self.n_outputs == 4:
        self.net = nn.Sequential(self.conv1, self.BatchNorm1, self.relu1, self.dropout1)
        #else:
        # self.net = nn.Sequential(self.conv1, self.BatchNorm1 ,self.relu1, self.dropout1,
        #                         self.conv2, self.BatchNorm2 ,self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None

        # self.relu = nn.LeakyReLU(negative_slope=0.5)
        self.relu = nn.ReLU()


        self.init_weights()

    def init_weights(self):

        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        # res1 = x if self.downsample is  None else self.downsample1(x)
        if self.n_outputs == 4:
            # output_res_n_x = out+res1
            output_res_n_x = out + res
            output_res_n_x = output_res_n_x[:,:,-(output_res_n_x.shape[2])//3:]
        else:
            output_res_n_x = self.relu(out+res)

        return output_res_n_x


class TCN(nn.Module):
    def __init__(self,config):# num_inputs, num_channels, kernel_size=2, dropout=0.2
        super(TCN, self).__init__()

        self.num_inputs=num_inputs=config.capacity
        self.num_channels=num_channels=[8,4]
        self.kernel_size=kernel_size=config.ks
        self.dropout=dropout=config.dropout



        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i  # 膨胀系数：1，2，4，8……
            in_channels = num_inputs if i == 0 else num_channels[i - 1]  # 确定每一层的输入通道数
            out_channels = num_channels[i]  # 确定每一层的输出通道数
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=dilation_size*(kernel_size-1), dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x=x.permute(0,2,1)



        x=self.network(x)
        x=x.permute(0,2,1)
        return x

class CausalConv1d(nn.Conv1d):
    def __init__(self,n_inputs, n_outputs, kernel_size,
                                           stride, padding, dilation):
        self.F_padding = padding
        self.stride = stride
        self.dilation = dilation
        self.padding = 0
        # if  n_outputs==4 :#对最后的输出长度进行修改，n_inputs是输出的通道数，下面这个值是输出长度为15
        #     self.padding = 0
        #     self.stride = 3
        #     self.dilation = 1
        #     self.F_padding = 0

        super(CausalConv1d, self).__init__(in_channels=n_inputs, out_channels=n_outputs, kernel_size=kernel_size,
                                           stride=self.stride, dilation=self.dilation,padding =self.padding )


    def forward(self, x):#这里是对因果卷积在膨胀卷积下依然成立做了处理

        x= F.pad(x,(self.F_padding,0))


        return super(CausalConv1d, self).forward(x)

