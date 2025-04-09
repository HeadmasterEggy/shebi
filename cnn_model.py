# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Config(object):

    """配置参数"""
    def __init__(self, dataset=None, embedding=None):
        self.model_name = 'TextCNN'
        if dataset:
            self.train_path = dataset + '/data/train.txt'                               # 训练集
            self.dev_path = dataset + '/data/val.txt'                                   # 验证集
            self.test_path = dataset + '/data/test.txt'                                 # 测试集
            self.class_list = [x.strip() for x in open(
                dataset + '/data/class.txt', encoding='utf-8').readlines()] if dataset else []  # 类别名单
            self.vocab_path = dataset + '/data/vocab.pkl'                               # 词表
            self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'       # 模型训练结果
            self.log_path = dataset + '/log/' + self.model_name
            self.embedding_pretrained = torch.tensor(
                np.load(dataset + '/data/' + embedding)["embeddings"].astype('float32'))\
                if embedding != 'random' and dataset and embedding else None            # 预训练词向量
        else:
            # Default values when no dataset is provided
            self.train_path = ''
            self.dev_path = ''
            self.test_path = ''
            self.class_list = []
            self.vocab_path = ''
            self.save_path = ''
            self.log_path = ''
            self.embedding_pretrained = None
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.dropout = 0.5                                              # 随机失活
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list) if self.class_list else 2  # 类别数
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.num_epochs = 20                                            # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3                                       # 学习率
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300           # 字向量维度
        self.filter_sizes = (2, 3, 4)                                   # 卷积核尺寸
        self.num_filters = 256                                          # 卷积核数量(channels数)


'''Convolutional Neural Networks for Sentence Classification'''


class TextCNN(nn.Module):
    def __init__(self, config):
        super(TextCNN, self).__init__()
        
        # Check if config is the Config class itself rather than an instance
        if isinstance(config, type) and config.__name__ == 'Config':
            config = Config()  # Create a default instance
        
        # Use a larger default vocabulary size to accommodate potential larger indices
        self.vocab_size = max(config.n_vocab, 50000) if config.n_vocab > 0 else 50000
        self.padding_idx = self.vocab_size - 1
        
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
            # Store the actual embedding size for reference
            self.actual_vocab_size = config.embedding_pretrained.size(0)
        else:
            self.embedding = nn.Embedding(self.vocab_size, config.embed, padding_idx=self.padding_idx)
            self.actual_vocab_size = self.vocab_size
            
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)
        self.config = config

        # Additional improvements could include:
        self.use_batch_norm = True
        if self.use_batch_norm:
            self.bn = nn.ModuleList([nn.BatchNorm1d(config.num_filters) for _ in config.filter_sizes])

    def conv_and_pool(self, x, conv, bn=None):
        x = conv(x).squeeze(3)  # Apply convolution
        if bn is not None and self.use_batch_norm:
            x = bn(x)  # Apply batch normalization
        x = F.relu(x)  # Apply activation
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def safe_embedding_input(self, x):
        """Ensure input indices are within the embedding range"""
        if torch.max(x).item() >= self.actual_vocab_size:
            # Clamp the indices to prevent out-of-range errors
            x = torch.clamp(x, 0, self.actual_vocab_size - 1)
        return x

    def forward(self, x):
        """
        Forward pass of the TextCNN model.
        
        Args:
            x: Input tensor or tuple. If tuple, the first element is used.
        """
        # Handle different input formats
        if isinstance(x, (tuple, list)):
            x = x[0]  # Extract the input tensor from tuple/list
            
        # Ensure input indices are within embedding range
        x = self.safe_embedding_input(x)
            
        # Apply embedding layer
        out = self.embedding(x)  # Shape: [batch_size, seq_len, embedding_dim]
        
        # Reshape for convolution: [batch_size, 1, seq_len, embedding_dim]
        out = out.unsqueeze(1)  # Add channel dimension
        
        # Apply convolutions and pooling
        if self.use_batch_norm:
            out = torch.cat([self.conv_and_pool(out, conv, bn) 
                           for conv, bn in zip(self.convs, self.bn)], 1)
        else:
            out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out