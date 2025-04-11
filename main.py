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
import argparse  # Add argparse import

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
from lstm_model import LSTM_attention, LSTMModel
from cnn_model import TextCNN

def train(train_dataloader, model, device, epoches, lr, patience):
    """训练模型函数

    参数：
        train_dataloader: 训练数据的DataLoader对象
        model: 待训练的模型
        device: 训练设备（如GPU或CPU）
        epoches: 训练轮数
        lr: 学习率
        patience: 早停耐心值，默认为5，当验证集准确率连续patience轮未提升时提前终止训练

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
    best_acc = 0
    counter = 0  # 初始化早停计数器

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
            counter = 0  # 重置早停计数器

            torch.save(model, model_path)
            print(f'最佳模型已保存，准确率为: {best_acc:.4f}%')
        else:
            counter += 1  # 增加早停计数器
            print(f'早停计数器: {counter}/{patience}')

        # 执行学习率调度
        scheduler.step()

        # 检查早停条件
        if counter >= patience:
            print(f'早停机制触发，在第{epoch+1}轮训练后停止')
            break

        # 最后保存所有训练日志
        history_df = pd.DataFrame(history)
        history_df.to_csv(f"train_log_{args.model.lower()}.csv", index=False)


if __name__ == "__main__":
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description='Text Classification Model Training')
    parser.add_argument('--model', type=str, default='cnn', 
                      choices=['bi_lstm_attention', 'bi_lstm', 'lstm_attention', 'lstm', 'cnn'],
                      help='选择使用的模型类型: bi_lstm_attention, bi_lstm, lstm_attention, lstm 或 cnn (默认: cnn)')
    args = parser.parse_args()

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
    train_dataloader = DataLoader(train_loader, batch_size=Config.lstm_batch_size, shuffle=True,
                                  num_workers=0)  # 注意：num_workers设置为0时速度较快

    val_loader = Data_set(val_array, val_label)
    val_dataloader = DataLoader(val_loader, batch_size=Config.lstm_batch_size, shuffle=True, num_workers=0)

    test_loader = Data_set(test_array, test_label)
    test_dataloader = DataLoader(test_loader, batch_size=Config.lstm_batch_size, shuffle=True, num_workers=0)

    # 构建模型（使用带注意力机制的 LSTM）
    bi_lstm_attention_model = LSTM_attention(
        Config.vocab_size,
        Config.embedding_dim,
        w2vec,
        Config.update_w2v,
        Config.hidden_dim,
        Config.num_layers,
        Config.drop_keep_prob,
        Config.n_class,
        Config.bidirectional_1,
    )

    # 初始化双向LSTM模型
    bi_lstm_model = LSTMModel(
        Config.vocab_size,
        Config.embedding_dim,
        w2vec,
        Config.update_w2v,
        Config.hidden_dim,
        Config.num_layers,
        Config.drop_keep_prob,
        Config.n_class,
        Config.bidirectional_1,
    )
    
    # 初始化LSTM_attention模型
    lstm_attention_model = LSTM_attention(
        Config.vocab_size,
        Config.embedding_dim,
        w2vec,
        Config.update_w2v,
        Config.hidden_dim,
        Config.num_layers,
        Config.drop_keep_prob,
        Config.n_class,
        Config.bidirectional_2,
    )
    
    # 初始化LSTM模型
    lstm_model = LSTMModel(
        Config.vocab_size,
        Config.embedding_dim,
        w2vec,
        Config.update_w2v,
        Config.hidden_dim,
        Config.num_layers,
        Config.drop_keep_prob,
        Config.n_class,
        Config.bidirectional_2,
    )

    # 正确初始化CNN模型
    cnn_model = TextCNN(
        Config.dropout,
        Config.require_improvement,
        Config.vocab_size,
        Config.cnn_batch_size,
        Config.pad_size,
        Config.filter_sizes,
        Config.num_filters,
        w2vec,  # 修正：传入预训练词向量w2vec而不是embedding_dim
        Config.embedding_dim,
        Config.n_class,
    )

    # 根据命令行参数选择模型
    if args.model == 'bi_lstm_attention':
        model = bi_lstm_attention_model
        print('使用 Bi-LSTM 注意力模型训练')
    elif args.model == 'bi_lstm':
        model = bi_lstm_model
        print('使用 Bi-LSTM 模型训练')
    elif args.model == 'lstm_attention':    
        model = lstm_attention_model
        print('使用 LSTM 注意力模型训练')
    elif args.model == 'lstm':
        model = lstm_model
        print('使用 LSTM 模型训练')
    elif args.model == 'cnn':
        model = cnn_model
        print('使用 CNN 模型训练')

    # 保存模型（根据模型名字保存，并添加缓存）
    model_filename = f"{args.model}_model_best.pkl"
    model_path = os.path.join(Config.model_dir, model_filename)

    # 训练模型
    train(train_dataloader, model=model, device=device, epoches=Config.n_epoch, lr=Config.lr, patience=10)

    # 保存模型（使用torch默认缓存机制）
    torch.save(model, model_path)

    print(f"模型已保存为: {model_path}")
