# coding: UTF-8

'''用于句子分类的卷积神经网络'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(
            self,
            dropout,
            vocab_size,
            pad_size,
            filter_sizes,
            num_filters,
            pretrained_weight,
            embedding_dim,
            n_class,
    ):
        super(TextCNN, self).__init__()
        self.dropout = dropout
        self.vocab_size = vocab_size
        self.pad_size = pad_size
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.pretrained_weight = pretrained_weight
        self.embedding_dim = embedding_dim
        self.n_class = n_class

        # 设置填充词索引
        self.padding_idx = self.vocab_size - 1

        # 使用预训练权重初始化嵌入层，并设置 padding_idx
        self.embedding = nn.Embedding.from_pretrained(
            pretrained_weight,
            freeze=False,
            padding_idx=self.padding_idx
        )

        # 嵌入层后的 Dropout
        self.embedding_dropout = nn.Dropout(p=self.dropout)

        # 卷积层定义
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, num_filters, (k, embedding_dim)) for k in filter_sizes]
        )

        # 批归一化
        self.bn = nn.BatchNorm1d(num_filters * len(filter_sizes))

        # Dropout 层
        self.dropout = nn.Dropout(dropout)

        # 全连接层
        self.fc = nn.Linear(num_filters * len(filter_sizes), n_class)

        # 权重初始化
        for conv in self.convs:
            nn.init.xavier_uniform_(conv.weight)
        nn.init.xavier_uniform_(self.fc.weight)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        """
        前向传播方法
        """
        out = self.embedding(x)
        out = self.embedding_dropout(out)  # 添加嵌入层 Dropout
        out = out.unsqueeze(1)  # 增加通道维度

        # 卷积 + 池化
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)

        # 批归一化 + Dropout
        out = self.bn(out)
        out = self.dropout(out)

        # 全连接层
        out = self.fc(out)
        return out
