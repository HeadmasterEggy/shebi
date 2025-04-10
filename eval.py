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
import hashlib
import json
import logging
import os
import pickle
from io import open
from pathlib import Path
from cnn_model import TextCNN
import numpy as np
import pandas as pd
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import Config
from data_Process import (
    build_word2id,
    build_id2word,
    prepare_data,
    build_word2vec,
    text_to_array_nolabel,
    Data_set,
)
from bilstm_model import LSTM_attention

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 添加安全全局类
torch.serialization.add_safe_globals([nn.Embedding, LSTM_attention])

from sklearn.metrics import f1_score, recall_score, confusion_matrix
import torch
import torch.nn as nn
import pandas as pd


class CacheManager:
    """缓存管理器，用于管理所有缓存操作"""

    def __init__(self, cache_dir="./cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def _get_cache_path(self, cache_name, params=None):
        """获取缓存文件路径"""
        if params:
            # 创建一个新的字典，只包含基本类型的参数，避免不稳定的引用类型
            stable_params = {}
            for key, value in params.items():
                # 只保留基本类型的参数（字符串、数字、布尔值等）
                if isinstance(value, (str, int, float, bool)) or value is None:
                    stable_params[key] = value
                else:
                    # 对于复杂类型，使用其类型名称和长度（如果可用）作为标识
                    try:
                        stable_params[key] = f"{type(value).__name__}_{len(value)}"
                    except:
                        stable_params[key] = f"{type(value).__name__}"
            
            # 使用稳定参数创建唯一的缓存标识
            params_str = json.dumps(stable_params, sort_keys=True)
            logging.info(f"缓存参数 ({cache_name}): {stable_params}")
            cache_id = hashlib.md5(params_str.encode()).hexdigest()
            cache_path = self.cache_dir / f"{cache_name}_{cache_id}.pkl"
            logging.info(f"缓存路径: {cache_path}")
            return cache_path
        return self.cache_dir / f"{cache_name}.pkl"

    def save(self, data, cache_name, params=None):
        """保存数据到缓存"""
        cache_path = self._get_cache_path(cache_name, params)
        logging.info(f"保存缓存到: {cache_path}")
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)

    def load(self, cache_name, params=None):
        """从缓存加载数据"""
        cache_path = self._get_cache_path(cache_name, params)
        if cache_path.exists():
            logging.info(f"从缓存加载: {cache_path}")
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
                return data
        logging.info(f"缓存不存在: {cache_path}")
        return None

    def exists(self, cache_name, params=None):
        """检查缓存是否存在"""
        return self._get_cache_path(cache_name, params).exists()


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
            'accuracy': [accuracy],
            'f1': [100 * f1],
            'recall': [100 * recall]
        })
        
        # 如果文件存在则清空内容后写入，不存在则创建新文件
        metrics_df.to_csv('metrics_log.csv', mode='w', header=True, index=False)

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

    # 禁用梯度计算，减少内存消耗
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
    with open(path, "r", encoding="utf-8") as file:
        texts = file.readlines()
    # 读取停用词
    # stopwords = []
    # with open("data/stopword.txt", "r", encoding="utf-8") as f:
    #      for line in f.readlines():
    #          stopwords.append(line.strip())

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


def initialize_data(cache_manager=None, force_reload=False):
    """
    初始化数据、字典和模型，支持缓存。

    Args:
        cache_manager: 缓存管理器实例
        force_reload: 是否强制重新加载数据
    """
    if cache_manager is None:
        cache_manager = CacheManager()

    # 定义缓存参数
    cache_params = {
        "model_name": Config.model_name,  # 添加模型名称到缓存参数
        "word2id_path": Config.word2id_path,
        "train_path": Config.train_path,
        "val_path": Config.val_path,
        "test_path": Config.test_path,
        "seq_length": Config.max_sen_len
    }

    if not force_reload and cache_manager.exists("processed_data", cache_params):
        logging.info(f"从缓存加载预处理数据 (模型: {Config.model_name})...")
        return cache_manager.load("processed_data", cache_params)

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
    test_dataloader = DataLoader(test_loader, batch_size=Config.batch_size, shuffle=True, num_workers=0)

    val_loader = Data_set(val_array, val_label)
    val_dataloader = DataLoader(val_loader, batch_size=Config.batch_size, shuffle=True, num_workers=0)

    processed_data = {
        "word2id": word2id,
        "test_dataloader": test_dataloader,
        "val_dataloader": val_dataloader,
        "train_array": train_array,
        "train_label": train_label
    }

    # 保存到缓存
    cache_manager.save(processed_data, "processed_data", cache_params)

    return processed_data


def initialize_model(w2vec, device, cache_manager=None):
    """
    初始化模型并加载最优模型或初始模型，支持缓存。
    """
    if cache_manager is None:
        cache_manager = CacheManager()

    # 检查是否有缓存的模型状态
    model_cache_params = {
        "model_name": Config.model_name,  # 添加模型名称到缓存参数
        "vocab_size": Config.vocab_size,
        "embedding_dim": Config.embedding_dim,
        "hidden_dim": Config.hidden_dim,
        "num_layers": Config.num_layers
    }

    lstm_model = LSTM_attention(
        Config.vocab_size,
        Config.embedding_dim,
        w2vec,
        Config.update_w2v,
        Config.hidden_dim,
        Config.num_layers,
        Config.drop_keep_prob,
        Config.n_class,
        Config.bidirectional,
    )
    cnn_model = TextCNN(Config)
    model = lstm_model if Config.model_name == "LSTM" else cnn_model
    
    logging.info(f"使用 {Config.model_name} 模型")
    
    # 选择适合的模型路径
    best_model_path = Config.lstm_best_model_path if Config.model_name == "LSTM" else Config.cnn_best_model_path
    
    # 尝试加载缓存的模型状态
    cache_key = f"{Config.model_name.lower()}_model_state"
    logging.info(f"尝试加载模型缓存 {cache_key}，参数: {model_cache_params}")
    cached_state = cache_manager.load(cache_key, model_cache_params)
    
    if cached_state is not None:
        logging.info(f"从缓存加载 {Config.model_name} 模型状态...")
        model.load_state_dict(cached_state)
    else:
        # 加载最佳模型
        logging.info(f"从 {best_model_path} 加载 {Config.model_name} 模型...")
        loaded_model = torch.load(best_model_path, weights_only=False)
        if isinstance(loaded_model, dict):
            model.load_state_dict(loaded_model)
        else:
            model.load_state_dict(loaded_model.state_dict())
        # 保存模型状态到缓存
        cache_manager.save(model.state_dict(), cache_key, model_cache_params)

    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="模型评估与预测脚本")
    parser.add_argument('--no-cache', action='store_true', help='禁用缓存，强制重新加载数据')
    parser.add_argument('--cache-dir', type=str, default='./cache', help='缓存目录路径')
    parser.add_argument('--model', type=str, choices=['lstm', 'cnn'], default='cnn', 
                        help='选择模型类型: LSTM 或 CNN')
    args = parser.parse_args()
    
    # 设置选择的模型名称
    Config.model_name = args.model
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"使用设备: {device}")

    # 初始化缓存管理器
    cache_manager = CacheManager(args.cache_dir)

    # 初始化数据
    processed_data = initialize_data(cache_manager, force_reload=args.no_cache)
    word2id = processed_data["word2id"]
    test_dataloader = processed_data["test_dataloader"]
    val_dataloader = processed_data["val_dataloader"]
    train_array = processed_data["train_array"]
    train_label = processed_data["train_label"]

    # 生成或加载 word2vec
    w2vec_cache_params = {
        "pre_word2vec_path": Config.pre_word2vec_path,
        "word2id_size": len(word2id),  # 添加word2id的大小作为参数，确保word2id变化时缓存也会更新
        "model_name": Config.model_name  # 添加模型名称，确保不同模型使用相同的word2vec缓存
    }
    logging.info(f"尝试加载word2vec缓存，参数: {w2vec_cache_params}")
    w2vec = cache_manager.load("w2vec", w2vec_cache_params)
    if w2vec is None:
        logging.info("未找到word2vec缓存，正在生成...")
        w2vec = build_word2vec(Config.pre_word2vec_path, word2id, None)
        w2vec = torch.from_numpy(w2vec).float()
        logging.info("保存word2vec到缓存")
        cache_manager.save(w2vec, "w2vec", w2vec_cache_params)
    else:
        logging.info("成功从缓存加载word2vec")
        logging.info(f"word2vec形状: {w2vec.shape}")

    # 初始化模型
    model = initialize_model(w2vec, device, cache_manager)

    # 测试阶段
    logging.info("开始测试模型...")
    test_accuracy(model, test_dataloader, device)

    # 验证阶段
    logging.info("开始验证模型...")
    val_accuracy(model, val_dataloader, device)

    # 预测阶段
    logging.info("开始进行预测...")
    pre(word2id, model, Config.max_sen_len, Config.pre_path, device)


if __name__ == "__main__":
    main()
