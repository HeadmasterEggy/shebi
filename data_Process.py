# !usr/bin/env python
# -*- coding:utf-8 -*-

"""
@Author: QiaoYi
@Date: 2025-03-27 10:08:13
@LastEditors: QiaoYi
@Description: DataProcess
@FilePath:
"""

from __future__ import unicode_literals, print_function, division

import logging
import re
from io import open
import jieba
import gensim
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from config import Config

# 设置日志格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Data_set(Dataset):
    def __init__(self, Data, Label):
        self.Data = Data
        if Label is not None:  # 考虑对测试集的使用
            self.Label = Label

    def __len__(self):
        return len(self.Data)

    def __getitem__(self, index):
        if self.Label is not None:
            data = torch.from_numpy(self.Data[index])
            label = torch.from_numpy(self.Label[index])
            return data, label
        else:
            data = torch.from_numpy(self.Data[index])
            return data


def data_preview(file_path):
    """
    @description: 预览原始数据集: 大小、描述信息等。
    @param {*}
    - file_path: str, 给定数据文件路径。
    @return {*}
    - df: DataFrame, 数据以 DataFrame 格式返回
    """
    try:
        data = []
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file.readlines():
                parts = line.strip().split()
                if len(parts) > 1:  # 确保每行有标签和文本
                    data.append([int(parts[0]), parts[1:]])

        df = pd.DataFrame(data, columns=["label", "text"])

        # 打印数据集的描述信息
        logging.info(f"{'*' * 20} 原始数据集描述 ({file_path.split('/')[-1]}) {'*' * 20}")
        logging.info(f"数据长度: {len(data)}")
        logging.info(f"数据预览:\n{df.head()}")
        logging.info(f"标签计数:\n{df['label'].value_counts()}")

        return df
    except Exception as e:
        logging.error(f"读取文件失败: {e}")
        return None


def stopwords_list():  # 创建停用词表
    stopwords = [
        line.strip()
        for line in open(Config.stopword_path, encoding="UTF-8").readlines()
    ]
    return stopwords


def build_word2id(file):
    """
    @description: build the dict of 'word2id'
    @param {*}
    - file: str, word2id保存地址
    @return {*}
    返回word2id的字典
    """
    stopwords = stopwords_list()  # 获取停用词
    word2id = {"_PAD_": 0}  # 初始化字典，包含PAD标记
    path = [Config.train_path, Config.val_path]

    for _path in path:
        with open(_path, encoding="utf-8") as f:
            for line in f.readlines():
                out_list = []
                # 分割文本为单词
                sp = line.strip().split()

                for word in sp[1:]:  # 假设第一列是标签，跳过
                    if word not in stopwords:  # 去除停用词

                        # 清洗单词
                        word_clean = clean_text(word)
                        # 如果word经过清洗后不为空，并且不为单字符
                        if word_clean and len(word_clean) > 1:
                            out_list.append(word_clean)

                # 将不重复的词加入字典
                for word in out_list:
                    if word not in word2id:
                        word2id[word] = len(word2id)

    # 将word2id字典保存到文件
    with open(file, "w", encoding="utf-8") as f:
        for w in word2id:
            f.write(w + "\t")
            f.write(str(word2id[w]))
            f.write("\n")

    return word2id


def build_id2word(word2id):
    """
    @description: 构建 id2word 字典，反向映射 word2id 字典
    @param {*}
    - word2id: dict, word2id 字典，包含词汇及其对应的 ID
    @return {*}
    返回 id2word 字典，包含 ID 和对应的单词
    """
    id2word = {id_: word for word, id_ in word2id.items()}
    return id2word


def build_word2vec(fname, word2id, save_to_path=None):
    """
    @description: 生成与词汇集对应的 word2vec 向量。
    @param {*}
    - fname: str, 预训练的 word2vec 模型文件路径，通常是从外部资源下载的文件。
    - word2id: dict, 语料库中包含的词汇及其对应的 ID（word2id 字典）。
    - save_to_path: str, 可选的保存路径，用于保存生成的词向量（如果提供路径，结果将保存到该路径）。
    @return {*}
    返回一个包含每个词汇对应的 word2vec 向量的矩阵，形式为 {id: word2vec}。
    """

    # 计算语料库中总共包含多少个词汇（word2id 字典中的最大 ID 值）
    n_words = max(word2id.values()) + 1

    # 加载预训练的 word2vec 模型（词向量模型）
    model = gensim.models.KeyedVectors.load_word2vec_format(fname, binary=True)

    # 创建一个随机初始化的词向量矩阵，大小为 [n_words, vector_size]（默认初始化为 -1 到 1 之间的随机值）
    word_vecs = np.random.uniform(-1.0, 1.0, [n_words, model.vector_size])

    # 为词汇集中的每个词赋予对应的词向量
    for word, word_id in word2id.items():
        try:
            # 尝试从预训练的 word2vec 模型中获取词向量
            word_vecs[word_id] = model[word]
        except KeyError:
            # 如果词不在预训练模型的词汇中，跳过该词
            pass

    # 如果提供了保存路径，保存词向量矩阵到文件
    if save_to_path:
        with open(save_to_path, "w", encoding="utf-8") as f:
            for vec in word_vecs:
                # 将每个词向量转换为字符串并写入文件
                vec_str = " ".join(map(str, vec))
                f.write(vec_str + "\n")

    # 返回生成的词向量矩阵
    return word_vecs

def clean_text(review):
    """
    对文本进行清洗：
      - 去除标签、链接、表情、标点、数字及重复字符
      - 去除英文字符
      - 去除多余空格
    """
    # 处理 NaN 值
    # ===================== 正则表达式预编译 =====================
    TAG_RE = re.compile(r'<[^>]+>')
    URL_RE = re.compile(r'http[s]?://(?:[a-zA-Z0-9\$\-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    EMOJI_RE = re.compile(r'[\U0001F300-\U0001F9FF]|[\u2600-\u26FF]|[\u2700-\u27BF]')
    PUNCT_RE = re.compile(r'[^\w\s\u4e00-\u9fff]')
    DIGITS_RE = re.compile(r'\d+')
    SPACE_RE = re.compile(r'\s+')
    # 新增：用于去除英文字符
    ENGLISH_RE = re.compile(r'\b[a-zA-Z]{2,}\b')  # 删除 2个以上英文字母的词

    if pd.isna(review):
        return ""

    # 将非字符串类型转换为字符串
    if not isinstance(review, str):
        review = str(review)
    review = TAG_RE.sub('', review)
    review = URL_RE.sub('', review)
    review = EMOJI_RE.sub('', review)
    review = PUNCT_RE.sub('', review)
    review = DIGITS_RE.sub('', review)
    review = ENGLISH_RE.sub('', review)  # 去除英文
    review = SPACE_RE.sub(' ', review).strip()
    return review


def tokenize(review, stopwords):
    """
    对单个文本进行分词，去除停用词和空白字符，返回以空格分隔的字符串。
    """
    words = jieba.cut(review, cut_all=False)  # 使用精确模式进行分词
    # 返回一个空格分隔的字符串
    return ' '.join([word for word in words if word not in stopwords and word.strip()])


def process_texts(review, stopwords):
    """
    对所有文本进行清洗和分词，并返回处理后的词列表。
    """
    logging.info("开始清洗文本...")
    cleaned_texts = [clean_text(text) for text in tqdm(review, desc="清洗文本")]

    logging.info("开始分词处理...")
    tokenized_texts = [tokenize(text, stopwords) for text in tqdm(cleaned_texts, desc="分词处理")]

    return tokenized_texts

def text_to_array(word2id, seq_length, path):
    """
    @description: 文本转为索引数字模式。将原始文本数据集中的每个词转为对应的索引数字，并返回句子和标签。
    @param {*}
    - word2id: dict, 语料文本中包含的词汇集，词到索引的映射。
    - seq_length: int, 序列的固定长度。若句子过长将被截断，若过短则用零填充。
    - path: str, 待处理的原始文本数据集路径。
    @return {*}
    返回两个值：
        - sentences_array: numpy.ndarray，形状为 (句子数, seq_length)，包含句子的索引数组。
        - label_array: list，标签列表，包含每个句子的标签。
    """
    sentences_array = []
    label_array = []

    # 读取文件并处理每一行
    with open(path, encoding="utf-8") as file:
        for line in file.readlines():
            # 分割每行，获取标签和文本部分
            parts = line.strip().split()
            label = int(parts[0])  # 第一部分是标签
            sentence = parts[1:]  # 其余部分是文本
  
            # 将文本转换为索引，未找到的词汇用0填充
            indexed_sentence = [word2id.get(word, 0) for word in sentence]

            # 进行序列长度填充或截断
            if len(indexed_sentence) < seq_length:
                indexed_sentence = [0] * (seq_length - len(indexed_sentence)) + indexed_sentence  # 前填0
            else:
                indexed_sentence = indexed_sentence[:seq_length]  # 截断多余部分

            sentences_array.append(indexed_sentence)
            label_array.append(label)

    # 转换为NumPy数组，返回句子和标签
    return np.array(sentences_array), label_array


def text_to_array_nolabel(word2id, seq_length, path):
    """
    @description: 文本转为索引数字模式 - 将原始文本（仅包括文本）里的每个词转为word2id对应的索引数字，并以数组返回。
    @param {*}
    - word2id: dict, 语料文本中包含的词汇集，词到索引的映射。
    - seq_length: int, 序列的限定长度。若句子过长将被截断，若过短则用零填充。
    - path: str, 待处理的原始文本数据集路径。
    @return {*}
    返回原始文本转化为索引数字数组后的数据集。
    """

    sentences_array = []

    # 读取文件并处理每一行
    with open(path, encoding="utf-8") as file:
        for line in file.readlines():
            words = line.strip().split()  # 分割句子成单词
            indexed_sentence = [word2id.get(word, 0) for word in words]  # 将每个单词转为索引

            # 填充或截断句子至指定长度
            if len(indexed_sentence) < seq_length:
                indexed_sentence = [0] * (seq_length - len(indexed_sentence)) + indexed_sentence  # 前填0
            else:
                indexed_sentence = indexed_sentence[:seq_length]  # 截断多余部分

            sentences_array.append(indexed_sentence)

    # 转换为NumPy数组并返回
    return np.array(sentences_array)


def to_categorical(y, num_classes=None):
    """
    @description: 将类别转化为one-hot编码
    @param {*}
    - y: list, 类别特征的列表
    - num_class: int, 类别个数
    @return {*}
    返回one-hot编码数组,shape:（len(y), num_classes）
    """
    y = np.array(y, dtype="int")
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def prepare_data(w2id, train_path, val_path, test_path, seq_length):
    """
    @description: 得到数字索引表示的句子和标签
    @param {*}
    - w2id: dict, 语料文本中包含的词汇集.
    - train_path: str, 训练数据集所在路径
    - val_path: str, 验证数据集所在路径
    - test_path: str, 测试数据集所在路径
    - seq_length: int, 序列的固定长度
    @return {*}
    - train_array: array, 训练集文本数组, shape:(len(train), seq_len)
    - train_label: array, 训练集标签数组, shape:(len(train), 1)
    - val_array: array, 验证集文本数组, shape:(len(val), seq_len)
    - val_label: array, 验证集标签数组, shape:(len(val), 1)
    - test_array: array, 测试集文本数组, shape:(len(test), seq_len)
    - test_label: array, 测试集标签数组, shape:(len(test), 1)
    """

    # 获取训练集、验证集、测试集的文本和标签
    train_array, train_label = text_to_array(w2id, seq_length=seq_length, path=train_path)
    val_array, val_label = text_to_array(w2id, seq_length=seq_length, path=val_path)
    test_array, test_label = text_to_array(w2id, seq_length=seq_length, path=test_path)

    # 处理标签为二维数组，形状为 (len, 1)
    train_label = np.array(train_label).reshape(-1, 1)
    val_label = np.array(val_label).reshape(-1, 1)
    test_label = np.array(test_label).reshape(-1, 1)

    return train_array, train_label, val_array, val_label, test_array, test_label


if __name__ == "__main__":
    # preview data
    train_df = data_preview(Config.train_path)
    test_df = data_preview(Config.test_path)
    val_df = data_preview(Config.val_path)

    # 建立word2id
    word2id = build_word2id(Config.word2id_path)

    # 建立id2word
    id2word = build_id2word(word2id)

    # 建立word2vec
    w2vec = build_word2vec(
        Config.pre_word2vec_path, word2id, Config.corpus_word2vec_path
    )

    # 得到句子id表示和标签
    (
        train_array,
        train_label,
        val_array,
        val_label,
        test_array,
        test_label,
    ) = prepare_data(
        word2id,
        train_path=Config.train_path,
        val_path=Config.val_path,
        test_path=Config.test_path,
        seq_length=Config.max_sen_len,
    )

    np.savetxt("./word2vec/train_data.txt", train_array, fmt="%d")
    np.savetxt("./word2vec/val_data.txt", val_array, fmt="%d")
    np.savetxt("./word2vec/test_data.txt", test_array, fmt="%d")
    np.savetxt("./word2vec/train_label.txt", train_label, fmt="%d")
    np.savetxt("./word2vec/val_label.txt", val_label, fmt="%d")
    np.savetxt("./word2vec/test_label.txt", test_label, fmt="%d")
    logging.info("数据处理完成。")
