# !usr/bin/env python3
# -*- coding:utf-8 -*-
"""全局配置。

所有路径均以本文件所在目录（项目根目录）为基准解析，因此在任何机器、
任何工作目录下启动都成立。历史版本里 main.py 直接写死了作者本机的
/Users/joey/PycharmProjects/shebi/... 绝对路径，换机器即抛 FileNotFoundError。
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _p(*parts):
    """把项目内的相对路径拼成绝对路径。"""
    return os.path.join(BASE_DIR, *parts)


class Config:
    # mutual
    base_dir = BASE_DIR
    model_dir = _p("model")
    stopword_path = _p("data", "stopword.txt")
    train_path = _p("data", "train.txt")
    val_path = _p("data", "val.txt")
    test_path = _p("data", "test.txt")
    pre_path = _p("data", "pre.txt")
    word2id_path = _p("word2vec", "word2id.txt")
    pre_word2vec_path = _p("word2vec", "wiki_word2vec_50.bin")
    corpus_word2vec_path = _p("word2vec", "word_vec.txt")
    cnn_best_model_path = _p("model", "cnn_model_best.pkl")
    lstm_best_model_path = _p("model", "lstm_model_best.pkl")
    bilstm_best_model_path = _p("model", "bilstm_model_best.pkl")
    lstm_attention_best_model_path = _p("model", "lstm_attention_model_best.pkl")
    bilstm_attention_best_model_path = _p("model", "bilstm_attention_model_best.pkl")

    # 训练进度 / 超参交换文件（Web 端与 main.py 之间通信）
    runtime_dir = _p("runtime")
    progress_path = _p("runtime", "progress.json")
    params_path = _p("runtime", "params.json")

    # 可选模型与默认模型
    model_choices = ("cnn", "lstm", "bilstm", "lstm_attention", "bilstm_attention")
    # 默认 cnn：修复 BatchNorm 缺陷后，在 6 轮对等训练下它的测试集准确率
    # (89.49%) 略高于 bilstm (89.25%)，且单条推理快约一倍、训练快约五倍。
    default_model = "cnn"

    n_class = 2  # 分类数：分别为pos和neg
    n_epoch = 10  # 训练迭代周期，即遍历整个训练样本的次数
    lr = 0.0001  # 学习率；若opt='adadelta'，则不需要定义学习率

    # 词表大小仅作参考：真实词表由 build_word2id 依据当前数据集产出（当前 60723）。
    # 模型构建时一律以词向量矩阵的行数为准，见 utils.create_model。
    vocab_size = 60723
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
    num_filters = 128  # 卷积核数量(channels数)

    # PAD 在 word2id 中的固定索引
    pad_idx = 0


os.makedirs(Config.runtime_dir, exist_ok=True)
os.makedirs(Config.model_dir, exist_ok=True)
