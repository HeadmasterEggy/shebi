# !usr/bin/env python3
# -*- coding:utf-8 -*-


class Config:
    # mutual
    model_dir = "./model"
    stopword_path = "./data/stopword.txt"
    train_path = "data/train.txt"
    val_path = "./data/val.txt"
    test_path = "./data/test.txt"
    pre_path = "./data/pre.txt"
    word2id_path = "./word2vec/word2id.txt"
    pre_word2vec_path = "./word2vec/wiki_word2vec_50.bin"
    corpus_word2vec_path = "./word2vec/word_vec.txt"
    cnn_best_model_path = "./model/cnn_model_best.pkl"
    lstm_best_model_path = "./model/lstm_model_best.pkl"
    bilstm_best_model_path = "./model/bilstm_model_best.pkl"
    lstm_attention_best_model_path = "./model/lstm_attention_model_best.pkl"
    bilstm_attention_best_model_path = "./model/bilstm_attention_model_best.pkl"
    n_class = 2  # 分类数：分别为pos和neg
    n_epoch = 100  # 训练迭代周期，即遍历整个训练样本的次数
    lr = 0.0001  # 学习率；若opt='adadelta'，则不需要定义学习率
    vocab_size = 54848  # 词汇量，与word2id中的词汇量一致
    embedding_dim = 50  # 词向量维度
    batch_size = 64  # 批处理尺寸
    dropout = 0.5  # 随机失活
    patience = 100  # 提前停止训练的耐心值

    # Bi-LSTM, LSTM, LSTM+Attention, Bi-LSTM+Attention
    update_w2v = True  # 是否在训练中更新w2v
    max_sen_len = 75  # 句子最大长度
    hidden_dim = 128  # 隐藏层节点数
    num_layers = 2  # LSTM层数
    bidirectional_1 = True  # 是否使用双向LSTM
    bidirectional_2 = False  # 是否使用双向LSTM

    # CNN
    require_improvement = 1000  # 若超过1000batch效果还没提升，则提前结束训练
    pad_size = 32  # 每句话处理成的长度(短填长切)
    filter_sizes = (2, 3, 4)  # 卷积核尺寸
    num_filters = 256  # 卷积核数量(channels数)
