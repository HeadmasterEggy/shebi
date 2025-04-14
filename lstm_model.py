#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
@Author: QiaoYi
@Date: 2022-02-18 19:12:58
@Version: 1.0
@LastEditors: QiaoYi
@LastEditTime: 2022-02-25 11:18:51
@Description: build model: LSTM, LSTM+attention
@FilePath: model.py
"""
from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMModel(nn.Module):
    """
    基础LSTM模型用于序列分类任务

    该模型包含以下组件:
    1. 预训练的词嵌入层
    2. LSTM编码器(可选双向)
    3. 两层全连接网络用于分类
    """

    def __init__(
            self,
            vocab_size,  # 词汇表大小
            embedding_dim,  # 词嵌入维度
            pretrained_weight,  # 预训练的词嵌入权重
            update_w2v,  # 是否更新词嵌入
            hidden_dim,  # LSTM隐藏层维度
            num_layers,  # LSTM层数
            dropout,  # Dropout保留概率
            n_class,  # 分类类别数
            bidirectional,  # 是否使用双向LSTM
            **kwargs
    ):
        """
        初始化LSTM模型

        参数:
        - vocab_size: 整数, 词汇表大小
        - embedding_dim: 整数, 词嵌入维度
        - pretrained_weight: 张量, 包含预训练的词嵌入权重
        - update_w2v: 布尔值, 是否在训练过程中更新词嵌入
        - hidden_dim: 整数, LSTM隐藏层维度
        - num_layers: 整数, LSTM层数
        - dropout: 浮点数, Dropout层的保留概率
        - n_class: 整数, 分类类别数
        - bidirectional: 布尔值, 是否使用双向LSTM
        """
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.n_class = n_class
        self.bidirectional = bidirectional

        # 初始化词嵌入层, 使用预训练权重
        self.embedding = nn.Embedding.from_pretrained(pretrained_weight)
        self.embedding.weight.requires_grad = update_w2v  # 控制是否微调词嵌入

        # 初始化LSTM编码器
        self.encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=num_layers,
            bidirectional=self.bidirectional,
            dropout=dropout,
        )

        # 初始化分类器(两层全连接网络)
        # 如果是双向LSTM, 输入维度需要乘以2
        if self.bidirectional:
            self.decoder1 = nn.Linear(hidden_dim * 2 * 2, hidden_dim)  # *4是因为我们连接了两个方向的首尾隐藏状态
            self.decoder2 = nn.Linear(hidden_dim, n_class)
        else:
            self.decoder1 = nn.Linear(hidden_dim * 2, hidden_dim)  # *2是因为我们连接了首尾隐藏状态
            self.decoder2 = nn.Linear(hidden_dim, n_class)

    def forward(self, inputs):
        """
        前向传播方法

        参数:
        - inputs: 形状为[batch_size, seq_length]的整数张量, 表示输入序列的词索引

        返回:
        - outputs: 形状为[batch_size, n_class]的张量, 表示每个类别的预测分数
        """
        embeddings = self.embedding(inputs)
        states, hidden = self.encoder(embeddings.permute(1, 0, 2))
        encoding = torch.cat([states[0], states[-1]], dim=1)
        outputs = F.relu(self.decoder1(encoding))  # 第一层全连接, 使用ReLU激活
        outputs = self.decoder2(outputs)  # 第二层全连接, 输出各类别的分数
        return outputs


class LSTM_attention(nn.Module):
    """
    带注意力机制的LSTM模型用于序列分类任务

    该模型扩展了基础LSTM, 添加了注意力机制以关注输入序列的重要部分:
    1. 预训练的词嵌入层
    2. LSTM编码器(可选双向)
    3. 自注意力机制层
    4. 两层全连接网络用于分类
    """

    def __init__(
            self,
            vocab_size,  # 词汇表大小
            embedding_dim,  # 词嵌入维度
            pretrained_weight,  # 预训练的词嵌入权重
            update_w2v,  # 是否更新词嵌入
            hidden_dim,  # LSTM隐藏层维度
            num_layers,  # LSTM层数
            dropout,  # Dropout保留概率
            n_class,  # 分类类别数
            bidirectional,  # 是否使用双向LSTM
            **kwargs
    ):
        """
        初始化带注意力机制的LSTM模型

        参数:
        - vocab_size: 整数, 词汇表大小
        - embedding_dim: 整数, 词嵌入维度
        - pretrained_weight: 张量, 包含预训练的词嵌入权重
        - update_w2v: 布尔值, 是否在训练过程中更新词嵌入
        - hidden_dim: 整数, LSTM隐藏层维度
        - num_layers: 整数, LSTM层数
        - dropout: 浮点数, Dropout层的保留概率
        - n_class: 整数, 分类类别数
        - bidirectional: 布尔值, 是否使用双向LSTM
        """
        super(LSTM_attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.n_class = n_class
        self.bidirectional = bidirectional

        # 初始化词嵌入层, 使用预训练权重
        self.embedding = nn.Embedding.from_pretrained(pretrained_weight)
        self.embedding.weight.requires_grad = update_w2v  # 控制是否微调词嵌入

        # 初始化LSTM编码器
        self.encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=num_layers,
            bidirectional=self.bidirectional,
            dropout=dropout,
        )

        # 初始化注意力机制的参数
        # 注意: nn.Parameter将张量转换为模型参数, 使其可以在训练过程中更新
        # weight_W: 用于变换LSTM隐藏状态的权重矩阵
        # weight_proj: 用于将变换后的隐藏状态映射到注意力分数的权重向量
        if self.bidirectional:
            self.weight_W = nn.Parameter(torch.Tensor(2 * hidden_dim, 2 * hidden_dim))
            self.weight_proj = nn.Parameter(torch.Tensor(2 * hidden_dim, 1))
        else:
            self.weight_W = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
            self.weight_proj = nn.Parameter(torch.Tensor(hidden_dim, 1))

        # 注意力分数的权重向量, 用于将隐藏状态映射到注意力分数

        # 初始化分类器(两层全连接网络)
        if self.bidirectional:
            self.decoder1 = nn.Linear(hidden_dim * 2, hidden_dim)
            self.decoder2 = nn.Linear(hidden_dim, n_class)
        else:
            self.decoder1 = nn.Linear(hidden_dim, hidden_dim)
            self.decoder2 = nn.Linear(hidden_dim, n_class)

        # 使用均匀分布初始化注意力权重
        # 初始化范围为[-0.1, 0.1], 这有助于稳定训练初期的梯度
        nn.init.uniform_(self.weight_W, -0.1, 0.1)
        nn.init.uniform_(self.weight_proj, -0.1, 0.1)

    def forward(self, inputs):
        """
        前向传播方法

        参数:
        - inputs: 形状为[batch_size, seq_length]的整数张量, 表示输入序列的词索引

        返回:
        - outputs: 形状为[batch_size, n_class]的张量, 表示每个类别的预测分数
        """
        # 步骤1: 将输入转换为词嵌入向量
        # 输入: [batch_size, seq_length] => 输出: [batch_size, seq_length, embedding_dim]
        embeddings = self.embedding(inputs)

        # 步骤2: 将嵌入向量输入LSTM进行编码
        # 注意: 这里的permute操作不同于基础LSTM模型, 保持了原始的维度顺序
        # 输入形状: [batch_size, seq_length, embedding_dim]
        states, hidden = self.encoder(embeddings.permute(0, 1, 2))

        # 步骤3: 应用注意力机制
        # 3.1: 通过非线性变换计算注意力的隐藏表示
        # u形状: [batch_size, seq_length, 2*hidden_dim]
        u = torch.tanh(torch.matmul(states, self.weight_W))

        # 3.2: 计算注意力分数
        # att形状: [batch_size, seq_length, 1]
        att = torch.matmul(u, self.weight_proj)

        # 3.3: 应用softmax得到注意力权重(归一化的分数)
        # att_score形状: [batch_size, seq_length, 1]
        att_score = F.softmax(att, dim=1)

        # 3.4: 加权求和得到上下文向量
        # scored_x形状: [batch_size, seq_length, 2*hidden_dim]
        scored_x = states * att_score

        # 3.5: 沿序列维度求和得到最终特征表示
        # encoding形状: [batch_size, 2*hidden_dim]
        encoding = torch.sum(scored_x, dim=1)

        # 步骤4: 通过两层全连接网络进行分类
        outputs = self.decoder1(encoding)  # 第一层全连接
        outputs = self.decoder2(outputs)  # 第二层全连接, 输出各类别的分数
        # 最终输出形状: [batch_size, n_class]

        return outputs
