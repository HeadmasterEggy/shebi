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
            padding_idx=0,
    ):
        super(TextCNN, self).__init__()
        self.dropout_p = dropout
        self.vocab_size = vocab_size
        self.pad_size = pad_size
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.embedding_dim = embedding_dim
        self.n_class = n_class

        # PAD 的真实索引由 build_word2id 决定，_PAD_ 固定为 0。
        # 旧实现用 vocab_size - 1 作为 padding_idx，等于把词表末尾一个真实词当成了填充符。
        self.padding_idx = padding_idx

        # 使用预训练权重初始化嵌入层
        self.embedding = nn.Embedding.from_pretrained(
            pretrained_weight,
            freeze=False,
            padding_idx=self.padding_idx
        )

        # 嵌入层后的 Dropout
        self.embedding_dropout = nn.Dropout(p=self.dropout_p)

        # 卷积层定义
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, num_filters, (k, embedding_dim)) for k in filter_sizes]
        )

        # 归一化层。
        #
        # 这里原本是 nn.BatchNorm1d(num_filters * len(filter_sizes))，存在一个致命缺陷：
        # BatchNorm 在 train() 下用当前 batch 统计量，在 eval() 下切换到 running stats。
        # 由于嵌入层参与训练（freeze=False），特征分布在训练全程持续漂移，
        # running_mean / running_var 始终追不上，导致 eval() 下特征被压平，
        # 分类头退化为常数分类器 —— 验证集 6334 条全部被判为同一类（acc 49.87%），
        # 而同一份权重改用 batch 统计量可达 76.07%。
        #
        # LayerNorm 在样本内部归一化，不维护 running stats，train / eval 行为完全一致，
        # 且不受推理时 batch 大小影响（线上单条请求 batch=1，BatchNorm 在此本就不适用）。
        self.norm = nn.LayerNorm(num_filters * len(filter_sizes))

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
        out = self.embedding_dropout(out)
        out = out.unsqueeze(1)  # 增加通道维度

        # 卷积 + 池化
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)

        # 归一化 + Dropout
        out = self.norm(out)
        out = self.dropout(out)

        # 全连接层
        out = self.fc(out)
        return out
