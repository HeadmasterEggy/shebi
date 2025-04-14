# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F

'''用于句子分类的卷积神经网络'''


class TextCNN(nn.Module):
    def __init__(
            self,
            dropout,
            require_improvement,
            vocab_size,
            batch_size,
            pad_size,
            filter_sizes,
            num_filters,
            pretrained_weight,
            embedding_dim,
            n_class,
    ):
        super(TextCNN, self).__init__()
        self.dropout = dropout
        self.require_improvement = require_improvement
        self.batch_size = batch_size
        self.pad_size = pad_size
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.pretrained_weight = pretrained_weight
        self.embedding_dim = embedding_dim
        self.n_class = n_class

        # 使用提供的词汇表大小
        self.vocab_size = vocab_size
        self.padding_idx = self.vocab_size - 1

        # 使用预训练权重初始化嵌入层
        self.embedding = nn.Embedding.from_pretrained(pretrained_weight, freeze=False)

        # 使用模型参数而不是直接使用Config值
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, self.num_filters, (k, self.embedding_dim)) for k in self.filter_sizes])
        self.dropout = nn.Dropout(self.dropout)
        self.fc = nn.Linear(self.num_filters * len(self.filter_sizes), self.n_class)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        """
        修改后的前向方法，可处理直接张量和元组/列表作为输入
        这使其与训练函数格式兼容
        """
        # 处理不同的输入格式，以兼容训练函数
        if isinstance(x, tuple) or isinstance(x, list):
            input_tensor = x[0]
        else:
            input_tensor = x

        out = self.embedding(input_tensor)
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out
