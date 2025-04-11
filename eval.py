# !usr/bin/env python3
# -*- coding:utf-8 -*-

"""
@Author: QiaoYi
@Date: 2025-03-27 10:07:59
@LastEditors: QiaoYi
@Description: model evaluation
@FilePath: eval.py
@LastEditTime: 2025-03-27 10:07:58
"""
from __future__ import unicode_literals, print_function, division

import argparse
import logging
import os

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, recall_score, confusion_matrix
from torch.utils.data import DataLoader

from cnn_model import TextCNN
from config import Config
from data_Process import (
    build_word2id,
    build_id2word,
    prepare_data,
    build_word2vec,
    text_to_array_nolabel,
    Data_set,
)
from lstm_model import LSTM_attention, LSTMModel
# 导入模型工具模块
from utils import initialize_model

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 添加安全全局类
torch.serialization.add_safe_globals([nn.Embedding, LSTM_attention, LSTMModel, TextCNN])


def val_accuracy(model, val_dataloader, device, criterion=nn.CrossEntropyLoss()):
    """
    计算验证集的准确率和其他评估指标（如F1分数、召回率和混淆矩阵）。

    参数:
        model (Object): 用于情感分析的模型（Seq2Seq）。
        val_dataloader (DataLoader): 验证集的DataLoader。
        device (str): 模型运行的设备 {"cpu", "cuda"}。
        criterion (Object): 损失函数，默认为nn.CrossEntropyLoss()。

    返回:
        float: 验证集的准确率。
    """
    # 设置模型为评估模式
    model.eval()

    # 初始化计算正确预测和总样本数的计数器
    correct_predictions = 0
    total_samples = 0
    total_loss = 0.0  # 累积的验证损失
    all_labels = []  # 保存所有标签
    all_preds = []  # 保存所有预测结果

    # 不计算梯度，减少内存消耗
    with torch.no_grad():
        # 遍历验证集
        for inputs, targets in val_dataloader:
            # 将数据移到指定设备
            inputs, targets = inputs.to(device).long(), targets.to(device).squeeze(1).long()

            # 前向传播，计算输出
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            # 预测结果
            _, predicted_labels = torch.max(outputs, 1)

            # 更新正确预测数和总样本数
            correct_predictions += (predicted_labels == targets).sum().item()
            total_samples += targets.size(0)

            # 收集标签和预测结果用于计算其他评估指标
            all_labels.extend(targets.cpu().numpy())
            all_preds.extend(predicted_labels.cpu().numpy())

        # 计算各项评估指标
        accuracy = 100 * correct_predictions / total_samples  # 准确率
        f1 = f1_score(all_labels, all_preds, average="weighted")  # F1-score
        recall = recall_score(all_labels, all_preds, average="micro")  # 召回率
        confusion_mat = confusion_matrix(all_labels, all_preds)  # 混淆矩阵

        # 保存评估指标到CSV文件
        metrics_df = pd.DataFrame({
            'timestamp': [pd.Timestamp.now()],
            'type': ['validation'],
            'model': [model_name],
            'accuracy': [accuracy],
            'f1': [100 * f1],
            'recall': [100 * recall]
        })

        # 检查文件是否存在，决定是否写入表头
        file_exists = os.path.isfile('metrics_log.csv')
        # 使用追加模式，保留之前的结果
        metrics_df.to_csv('metrics_log.csv', mode='a', header=not file_exists, index=False)

        # 打印结果
        logging.info(f"\n验证集准确率: {accuracy:.3f}%, 总损失: {total_loss:.3f}, F1分数: {100 * f1:.3f}%, "
                     f"召回率: {100 * recall:.3f}%, 混淆矩阵:\n{confusion_mat}")

    return accuracy


def test_accuracy(model, test_dataloader, device):
    """
    计算测试集的准确率和其他评估指标（如F1分数、召回率和混淆矩阵）。

    参数:
        model (Object): 用于情感分析的模型（Seq2Seq）。
        test_dataloader (DataLoader): 测试集的DataLoader。
        device (str): 模型运行的设备 {"cpu", "cuda"}。

    返回:
        float: 测试集的准确率。
    """
    # 设置模型为评估模式
    model.eval()

    # 初始化正确预测数、总样本数以及累积的损失
    correct_predictions = 0
    total_samples = 0
    all_labels = []  # 保存所有标签
    all_preds = []  # 保存所有预测结果

    # 禁用梯度计算
    with torch.no_grad():
        # 遍历测试集数据
        for inputs, targets in test_dataloader:
            # 将数据移到指定设备
            inputs, targets = inputs.to(device).long(), targets.to(device).squeeze(1).long()

            # 前向传播，计算输出
            outputs = model(inputs)

            # 获取预测标签
            _, predicted_labels = torch.max(outputs, 1)

            # 更新正确预测数和总样本数
            correct_predictions += (predicted_labels == targets).sum().item()
            total_samples += targets.size(0)

            # 收集标签和预测结果用于计算其他评估指标
            all_labels.extend(targets.cpu().numpy())
            all_preds.extend(predicted_labels.cpu().numpy())

        # 计算各项评估指标
        accuracy = 100 * correct_predictions / total_samples  # 准确率
        f1 = f1_score(all_labels, all_preds, average="weighted")  # F1-score
        recall = recall_score(all_labels, all_preds, average="micro")  # 召回率
        confusion_mat = confusion_matrix(all_labels, all_preds)  # 混淆矩阵

        # 保存评估指标到CSV文件
        metrics_df = pd.DataFrame({
            'timestamp': [pd.Timestamp.now()],
            'type': ['test'],
            'model': [model_name],
            'accuracy': [accuracy],
            'f1': [100 * f1],
            'recall': [100 * recall]
        })

        # 检查文件是否存在，决定是否写入表头
        file_exists = os.path.isfile('metrics_log.csv')
        # 使用追加模式，保留之前的结果
        metrics_df.to_csv('metrics_log.csv', mode='a', header=not file_exists, index=False)

        # 使用 logging 记录结果
        logging.info(f"\n测试集准确率: {accuracy:.3f}%, F1分数: {100 * f1:.3f}%, "
                     f"召回率: {100 * recall:.3f}%, 混淆矩阵:\n{confusion_mat}")

    return accuracy


def pre(word2id, model, seq_length, path, device=None):
    """
    给定文本，预测其情感标签。

    参数:
        word2id (dict): 语料文本中包含的词汇字典（词和ID映射关系）。
        model (Object): 情感分析模型（Seq2Seq）。
        seq_length (int): 序列的固定长度。
        path (str): 数据文件路径。
        device (torch.device, optional): 运行设备，默认为None（使用CPU）。

    返回:
        list: 预测的情感标签（0 或 1）。
    """
    # 如果没有指定设备，默认使用CPU
    if device is None:
        device = torch.device("cpu")
        model = model.cpu()
    else:
        model = model.to(device)

    model.eval()  # 确保模型处于评估模式

    # 读取文件中的文本
    with open(path, "r", encoding="utf-8") as file, open("data/stopword.txt", "r", encoding="utf-8") as f:
        texts = file.readlines()
        stopwords = [line.strip() for line in f.readlines()]

    # texts = process_texts(texts, stopwords)
    predictions = []  # 用于存储预测的标签

    with torch.no_grad():  # 禁用梯度计算
        # 将文本转换为索引数字
        input_array = text_to_array_nolabel(word2id, seq_length, path)
        sen_p = torch.tensor(input_array, dtype=torch.long).to(device)

        # 获取模型预测的输出
        output_p = model(sen_p)
        # 获取每个文本的预测类别
        _, pred = torch.max(output_p, 1)

        for i in range(pred.size(0)):
            logits = output_p[i].tolist()
            prediction = pred[i].item()
            predictions.append(prediction)
            probs = F.softmax(output_p, dim=1)

            sentiment = '负面' if prediction == 0 else '正面'
            prob = probs[i].tolist()
            confidence = max(prob)
            logging.info(f" - 预测类别: {sentiment} ({prediction}), 置信度: {confidence:.4f}")
            # 使用日志记录预测结果
            logging.info(f"文本: {texts[i].strip()} => 预测类别: {sentiment}")

    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="模型评估与预测脚本")
    parser.add_argument('--model', type=str, default='cnn',
                        choices=['bilstm_attention', 'bilstm', 'lstm_attention', 'lstm', 'cnn'],
                        help='选择使用的模型类型: BiLSTM_attention, BiLSTM, LSTM_attention, LSTM 或 TextCNN (默认: TextCNN)')
    args = parser.parse_args()
    
    model_name = args.model.lower()  # 统一转为小写处理
    logging.info(f"模型名称: {model_name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"使用设备: {device}")

    # 初始化数据
    logging.info("初始化数据...")
    word2id = build_word2id(Config.word2id_path)
    id2word = build_id2word(word2id)

    # 准备训练、验证和测试数据
    train_array, train_label, val_array, val_label, test_array, test_label = prepare_data(
        word2id,
        train_path=Config.train_path,
        val_path=Config.val_path,
        test_path=Config.test_path,
        seq_length=Config.max_sen_len,
    )

    # 创建 DataLoader
    test_loader = Data_set(test_array, test_label)
    test_dataloader = DataLoader(test_loader, batch_size=Config.lstm_batch_size, shuffle=True, num_workers=0)

    val_loader = Data_set(val_array, val_label)
    val_dataloader = DataLoader(val_loader, batch_size=Config.lstm_batch_size, shuffle=True, num_workers=0)

    # 生成 word2vec
    logging.info("生成word2vec...")
    w2vec = build_word2vec(Config.pre_word2vec_path, word2id, None)
    w2vec = torch.from_numpy(w2vec).float()

    # 使用导入的初始化模型函数，传入设备
    model = initialize_model(model_name, w2vec, device)
    logging.info("模型初始化完成")
    
    # 测试阶段
    logging.info("开始测试模型...")
    test_accuracy(model, test_dataloader, device)

    # 验证阶段
    logging.info("开始验证模型...")
    val_accuracy(model, val_dataloader, device)

    # 预测阶段
    logging.info("开始进行预测...")
    pre(word2id, model, Config.max_sen_len, Config.pre_path, device)