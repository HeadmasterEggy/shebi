"""
@Author: QiaoYi
@Date: 2025-03-27 10:07:59
@LastEditors: QiaoYi
@LastEditTime: 2025-03-27 10:07:59
@Description: main
@FilePath: main.py
"""
from __future__ import unicode_literals, print_function, division

import os

import pandas as pd
import torch
import torch.nn as nn
import tqdm
from sklearn.metrics import f1_score, recall_score
from torch import optim
from torch.utils.data import DataLoader

from config import Config
from data_Process import (
    data_preview,
    prepare_data,
    build_word2id,
    build_id2word,
    build_word2vec,
    Data_set,
)
from eval import val_accuracy
from model import LSTM_attention


def train(train_dataloader, model, device, epoches, lr):
    """训练模型函数

    参数：
        train_dataloader: 训练数据的DataLoader对象
        model: 待训练的模型
        device: 训练设备（如GPU或CPU）
        epoches: 训练轮数
        lr: 学习率

    返回：
        无
    """
    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "f1": [],
        "recall": []
    }
    model.train()
    model = model.to(device)
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    # 学习率调整：每10轮降低学习率，衰减系数为0.2
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)
    best_acc = 0.85
    for epoch in range(epoches):
        train_loss = 0.0
        correct = 0
        total = 0

        train_dataloader_cur = tqdm.tqdm(train_dataloader)
        train_dataloader_cur.set_description(
            '[Epoch: {:04d}/{:04d} lr: {:.6f}]'.format(epoch + 1, epoches, scheduler.get_last_lr()[0]))
        for i, data_ in enumerate(train_dataloader_cur):
            # 清空梯度
            optimizer.zero_grad()
            input_, target = data_[0], data_[1]
            input_ = input_.type(torch.LongTensor)
            target = target.type(torch.LongTensor)
            # 将数据移动到指定设备
            input_ = input_.to(device)
            target = target.to(device)

            # 模型前向传播，输出形状: [num_samples, 类别数]
            output = model(input_)

            # 调整目标标签形状：将 [num_samples, 1] 转为 [num_samples]
            target = target.squeeze(1)

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            # 获取预测标签，返回值中第二个为预测的类别索引
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            F1 = f1_score(target.cpu(), predicted.cpu(), average="weighted")
            Recall = recall_score(target.cpu(), predicted.cpu(), average="micro")

            postfix = {
                "train_loss: {:.5f}, train_acc: {:.3f}%, F1: {:.3f}%, Recall: {:.3f}%".format(train_loss / (i + 1),
                                                                                              100 * correct / total,
                                                                                              100 * F1, 100 * Recall)}
            train_dataloader_cur.set_postfix(log=postfix)

            # 计算 epoch 平均指标
            avg_loss = train_loss / len(train_dataloader)
            avg_acc = 100 * correct / total
            f1_percent = 100 * F1
            recall_percent = 100 * Recall

            # 保存日志
            history["epoch"].append(epoch + 1)
            history["train_loss"].append(avg_loss)
            history["train_acc"].append(avg_acc)
            history["f1"].append(f1_percent)
            history["recall"].append(recall_percent)

        # 注意：val_dataloader 为全局变量，在 __main__ 中定义，用于模型验证
        acc = val_accuracy(model, val_dataloader, device, criterion)
        model.train()

        if acc > best_acc:
            best_acc = acc
            if not os.path.exists(Config.model_dir):
                os.mkdir(Config.model_dir)
            torch.save(model, Config.best_model_path)
        # 最后保存所有训练日志
        history_df = pd.DataFrame(history)
        history_df.to_csv("train_log.csv", index=False)


if __name__ == "__main__":
    # 主函数：预览数据、预处理、模型构建、训练和保存模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据预览，观察训练、测试和验证数据概况
    train_df = data_preview(Config.train_path)
    test_df = data_preview(Config.test_path)
    val_df = data_preview(Config.val_path)

    # 构建词表映射：建立 word2id 和 id2word 对应关系
    word2id = build_word2id(Config.word2id_path)
    id2word = build_id2word(word2id)

    # 数据预处理：生成句子表示及其对应的标签
    (train_array, train_label, val_array, val_label, test_array, test_label) = prepare_data(
        word2id,
        train_path=Config.train_path,
        val_path=Config.val_path,
        test_path=Config.test_path,
        seq_length=Config.max_sen_len,
    )

    # 生成 word2vec 向量，并转换为 float32 类型（适用于 CUDA 环境）
    w2vec = build_word2vec(Config.pre_word2vec_path, word2id, None)
    w2vec = torch.from_numpy(w2vec).float()

    # 构建数据加载器
    train_loader = Data_set(train_array, train_label)
    train_dataloader = DataLoader(train_loader, batch_size=Config.batch_size, shuffle=True,
                                  num_workers=0)  # 注意：num_workers设置为0时速度较快

    val_loader = Data_set(val_array, val_label)
    val_dataloader = DataLoader(val_loader, batch_size=Config.batch_size, shuffle=True, num_workers=0)

    test_loader = Data_set(test_array, test_label)
    test_dataloader = DataLoader(test_loader, batch_size=Config.batch_size, shuffle=True, num_workers=0)

    # 构建模型（使用带注意力机制的 LSTM）
    model = LSTM_attention(
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

    # 训练模型
    train(train_dataloader, model=model, device=device, epoches=Config.n_epoch, lr=Config.lr)

    # 保存模型（若模型存储目录不存在则创建目录）
    if not os.path.exists(Config.model_dir):
        os.mkdir(Config.model_dir)
    torch.save(model, Config.model_state_dict_path)
