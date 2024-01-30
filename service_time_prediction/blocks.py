import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CasualConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, dilation=1, groups=1, bias=True):
        super(CasualConv1d, self).__init__()
        self.dilation = dilation
        padding = dilation * (kernel_size - 1)
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride,
                                padding, dilation, groups, bias)

    def forward(self, input):
        # Takes something of shape (N, in_channels, T),
        # returns (N, out_channels, T)
        out = self.conv1d(input)
        return out[:, :, :-self.dilation]  # TODO: make this correct for different strides/padding


class DenseBlock(nn.Module):
    def __init__(self, in_channels, dilation, filters, kernel_size=2):
        super(DenseBlock, self).__init__()
        self.casualconv1 = CasualConv1d(in_channels, filters, kernel_size, dilation=dilation)
        self.casualconv2 = CasualConv1d(in_channels, filters, kernel_size, dilation=dilation)

    def forward(self, input):
        # input is dimensions (N, in_channels, T)
        xf = self.casualconv1(input)
        xg = self.casualconv2(input)
        activations = torch.tanh(xf) * torch.sigmoid(xg)  # shape: (N, filters, T)
        return torch.cat((input, activations), dim=1)  # [N, in+filters, T]


class TCBlock(nn.Module):
    def __init__(self, in_channels, seq_length, filters, dev=None):
        super(TCBlock, self).__init__()
        self.device = dev if dev else torch.device('cpu')
        # 叠加不同stride的空洞卷积 每次filter个数都是一样的
        self.dense_blocks = nn.ModuleList([DenseBlock(in_channels + i * filters, 2 ** (i + 1), filters)
                                           for i in range(int(math.ceil(math.log(seq_length, 2))))]).to(
            device=self.device)

    def forward(self, input):
        # input is dimensions (N, T, in_channels)
        input = torch.transpose(input, 1, 2)
        for block in self.dense_blocks:
            input = block(input)
        return torch.transpose(input, 1, 2)


class SelfAttentionBlock(nn.Module):
    def __init__(self, in_channels, key_size, value_size, dev, att_dim, mask_f=True):
        super(SelfAttentionBlock, self).__init__()
        self.dev = dev
        self.linear_query = nn.Linear(in_channels, key_size, device=self.dev)
        self.linear_keys = nn.Linear(in_channels, key_size, device=self.dev)
        self.linear_values = nn.Linear(in_channels, value_size, device=self.dev)
        self.sqrt_key_size = math.sqrt(key_size)
        self.att_dim=att_dim
        self.mask_f=mask_f

    def forward(self, input):
        # input is dim (N, T, in_channels) where N is the batch_size, and T is
        # the sequence length
        mask = np.array([[1 if i > j else 0 for i in range(input.shape[1])] for j in range(input.shape[1])])
        mask = torch.BoolTensor(mask).to(self.dev)

        # import pdb; pdb.set_trace()
        keys = self.linear_keys(input)  # shape: (N, T, key_size)
        query = self.linear_query(input)  # shape: (N, T, key_size)
        values = self.linear_values(input)  # shape: (N, T, value_size)
        temp = torch.bmm(query, torch.transpose(keys, 1, 2))  # shape: (N, T, T)
        if self.mask_f:
            temp.data.masked_fill_(mask, -float('inf'))
        temp = F.softmax(temp / self.sqrt_key_size,
                         dim=self.att_dim)    # 改成2是改正后试一下 # shape: (N, T, T), broadcasting over any slice [:, x, :], each row of the matrix
        temp = torch.bmm(temp, values)  # shape: (N, T, value_size)
        return torch.cat((input, temp), dim=2)  # shape: (N, T, in_channels + value_size)


class QSAttentionBlock1(nn.Module):
    def __init__(self, in_channels_key, in_channels_query, key_size, value_size, dev):
        super(QSAttentionBlock1, self).__init__()
        self.dev = dev

        self.linear_keys = nn.Linear(in_channels_key, key_size, device=self.dev)
        self.linear_query = nn.Linear(in_channels_query, key_size, device=self.dev)
        self.linear_values_spt = nn.Linear(in_channels_key, value_size, device=self.dev)
        self.fc = nn.Linear(in_channels_query, in_channels_key)
        self.sqrt_key_size = math.sqrt(key_size)
        self.value_size = value_size

    def forward(self, spt, qry):


        # import pdb; pdb.set_trace()
        if len(spt) == 0:

            # 不对称
            x = torch.cat([self.fc(qry), qry[:, :self.value_size]], 1)
            x = torch.unsqueeze(x, 1)
            return x


        qry_size = len(qry)
        keys = self.linear_keys(spt)  # shape: (T, key_size)
        query = self.linear_query(qry)  # shape: (Q, key_size)
        values = self.linear_values_spt(spt)  # shape: (T, value_size)
        # print("keys.size=", keys.size())
        temp = torch.mm(query, torch.transpose(keys, 0, 1))  # shape: (Q, T)
        # temp.data.masked_fill_(mask, -float('inf'))
        weights = F.softmax(temp / self.sqrt_key_size,
                            dim=1)    # 本来就应该是1 # shape: (Q, T), broadcasting over any slice [:, x, :], each row of
        # the matrix
        qry_values = torch.mm(weights, values)  # shape: (Q, value_size)
        weights_exp = torch.unsqueeze(weights, 2).expand(-1, -1, self.value_size)
        spt_values = torch.multiply(values.expand(qry_size, -1, -1), weights_exp)  #
        # shape: (Q, T, value_size)


        qry_values = torch.cat([self.fc(qry), qry_values], dim=1)  # shape: (Q, in_channels_key + value_size)
        x = torch.cat([spt.expand(qry_size, -1, -1), spt_values], 2)  # shape: (Q, T, in_channels_key + value_size)
        x = torch.cat([x, torch.unsqueeze(qry_values, 1)], 1)  # shape: (Q, T + 1, in_channels_key + value_size)
        return x


class QSAttentionBlock2(nn.Module):
    def __init__(self, in_channels, key_size, value_size, dev,att_dim):
        super(QSAttentionBlock2, self).__init__()
        self.dev = dev
        self.linear_query = nn.Linear(in_channels, key_size, device=self.dev)
        self.linear_keys = nn.Linear(in_channels, key_size, device=self.dev)
        self.linear_values = nn.Linear(in_channels, value_size, device=self.dev)
        self.sqrt_key_size = math.sqrt(key_size)
        self.att_dim=att_dim

    def forward(self, input):
        # input is dim (Q, S+1, in_channels) where N is the batch_size, and T is
        # the sequence length
        mask = np.array([[1 if i > j else 0 for i in range(input.shape[1])] for j in range(input.shape[1])])
        mask = torch.BoolTensor(mask).to(self.dev)

        # import pdb; pdb.set_trace()
        keys = self.linear_keys(input)  # shape: (Q, S+1, key_size)
        query = self.linear_query(input)  # shape: (Q, S+1, key_size)
        values = self.linear_values(input)  # shape: (Q, S+1, value_size)
        temp = torch.bmm(query, torch.transpose(keys, 1, 2))  # shape: (Q, S+1, S+1)
        temp.data.masked_fill_(mask, -float('inf'))
        temp = F.softmax(temp / self.sqrt_key_size,
                         dim=self.att_dim)    # 改成2是改正后试一下 # shape: (Q, S+1, S+1), broadcasting over any slice [:, x,
        # :], each row of the matrix
        temp = torch.bmm(temp, values)  # shape: (Q, S+1, value_size)
        return torch.cat((input, temp), dim=2)  # shape: (Q, S+1, in_channels + value_size)


class SpatialAttentionBlock(nn.Module):
    def __init__(self, in_channels, key_size, value_size, feature_dim, dev):
        super(SpatialAttentionBlock, self).__init__()
        self.dev = dev
        self.linear_query = nn.Linear(in_channels + feature_dim, key_size, device=self.dev)
        self.linear_keys = nn.Linear(in_channels, key_size, device=self.dev)
        self.linear_values = nn.Linear(in_channels, value_size, device=self.dev)
        self.sqrt_key_size = math.sqrt(key_size)
        # self.dev = dev

    def forward(self, input, feature):
        # input is dim (N, T, in_channels) where N is the batch_size, and T is
        # (N, T, feature_dim)
        # the sequence length
        mask = np.array([[1 if i > j else 0 for i in range(input.shape[1])] for j in range(input.shape[1])])
        mask = torch.BoolTensor(mask).to(self.dev)

        # import pdb; pdb.set_trace()
        keys = self.linear_keys(input)  # shape: (N, T, key_size)
        # query = self.linear_query(input) # shape: (N, T, key_size)
        query = self.linear_query(torch.cat([input, feature], dim=2))  # shape: (N, T, key_size)
        values = self.linear_values(input)  # shape: (N, T, value_size)
        temp = torch.bmm(query, torch.transpose(keys, 1, 2))  # shape: (N, T, T)
        temp.data.masked_fill_(mask, -float('inf'))
        temp = F.softmax(temp / self.sqrt_key_size,
                         dim=1)  # shape: (N, T, T), broadcasting over any slice [:, x, :], each row of the matrix
        temp = torch.bmm(temp, values)  # shape: (N, T, value_size)
        return torch.cat((input, temp), dim=2)  # shape: (N, T, in_channels + value_size)
