import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):

        support = torch.mm(input, self.weight)

        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class SelfAttend(nn.Module):
    def __init__(self, embedding_size: int) -> None:
        super(SelfAttend, self).__init__()

        self.h1 = nn.Sequential(
            nn.Linear(embedding_size, 32),  # 将输入的embedding_size映射到大小为32的隐藏层
            nn.Tanh()  # 激活函数非线性变换
        )
        
        self.gate_layer = nn.Linear(32, 1)  # 将隐藏层的输出映射到单一的输出维度，表示注意力权重

    def forward(self, seqs, seq_masks=None):
        """
        :param seqs: shape [batch_size, seq_length, embedding_size]
        :param seq_lens: shape [batch_size, seq_length]
        :return: shape [batch_size, seq_length, embedding_size]
        """
        gates = self.gate_layer(self.h1(seqs)).squeeze(-1)  # 注意力权重
        if seq_masks is not None:
            gates = gates + seq_masks  # 如果提供掩码序列，则将其与注意力权重相加
        p_attn = F.softmax(gates, dim=-1)  # 注意力权重归一化
        p_attn = p_attn.unsqueeze(-1)
        h = seqs * p_attn
        # output = torch.sum(h, dim=1)  # 加权和输出
        output = torch.sum(h, dim=1).unsqueeze(dim=0)  # 加权和输出
        return output

