"""
@Author: QiaoYi
@Date: 2025-03-27 10:07:59
@LastEditors: QiaoYi
@LastEditTime: 2025-03-27 10:07:59
@Description: main
@FilePath: main.py
"""
from __future__ import unicode_literals, print_function, division

import argparse
import json
import os
import random
import sys
import traceback

import numpy as np
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
# 导入模型工具模块
from utils import create_model


def set_seed(seed):
    """
    设置随机种子，确保实验的可重复性

    参数:
        seed (int): 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f'设置随机种子为: {seed}')


def train(train_dataloader, model, device, epoches, lr, patience):
    """训练模型函数

    参数：
        train_dataloader: 训练数据的DataLoader对象
        model: 待训练的模型
        device: 训练设备（如GPU或CPU）
        epoches: 训练轮数
        lr: 学习率
        patience: 早停耐心值，默认为10，当验证集准确率连续patience轮未提升时提前终止训练

    返回：
        无
    """
    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "f1": [],
        "recall": [],
        "val_acc": [],  # 添加验证准确率记录
        "val_loss": []  # 添加验证损失记录
    }
    model.train()
    model = model.to(device)
    print(model)

    # 更新训练进度
    update_training_progress(0, epoches, None, None, None, 'running', "训练开始")

    # 定义优化器和损失函数
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    # 余弦退火学习率调整：在每个周期内学习率从初始值余弦衰减到最小值
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epoches,  # 一个完整的余弦周期的长度，设为总轮数
        eta_min=1e-6  # 最小学习率
    )

    best_acc = 0
    counter = 0  # 初始化早停计数器

    for epoch in range(epoches):
        train_loss = 0.0
        correct = 0
        total = 0
        all_targets = []  # 收集所有的真实标签
        all_predictions = []  # 收集所有的预测结果

        train_dataloader_cur = tqdm.tqdm(train_dataloader)
        train_dataloader_cur.set_description(
            '[Epoch: {:04d}/{:04d} lr: {:.6f}]'.format(epoch + 1, epoches, scheduler.get_last_lr()[0]))
        for i, data_ in enumerate(train_dataloader_cur):
            try:
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
                                                                                                  100 * F1,
                                                                                                  100 * Recall)}

                # 收集用于计算整个epoch的F1和召回率
                all_targets.extend(target.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

                train_dataloader_cur.set_postfix(log=postfix)

            except Exception as e:
                print(f"批次 {i} 处理出错: {str(e)}")
                traceback.print_exc()
                update_training_progress(
                    epoch + 1, epoches, train_loss / (i + 1) if i > 0 else None,
                    None, None, 'failed', f"训练批次出错: {str(e)}"
                )
                continue

        # 计算整个epoch的指标
        avg_loss = train_loss / len(train_dataloader)
        avg_acc = 100 * correct / total
        F1 = f1_score(all_targets, all_predictions, average="weighted")
        Recall = recall_score(all_targets, all_predictions, average="micro")
        f1_percent = 100 * F1
        recall_percent = 100 * Recall

        # 每个epoch结束后，将该epoch的指标添加到history中
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(avg_loss)
        history["train_acc"].append(avg_acc)
        history["f1"].append(f1_percent)
        history["recall"].append(recall_percent)

        try:
            # 注意：val_dataloader 为全局变量，在 __main__ 中定义，用于模型验证
            acc, val_loss = val_accuracy(model, val_dataloader, device, criterion)
            # 记录验证集指标
            history["val_acc"].append(acc)
            history["val_loss"].append(val_loss)
            model.train()

            # 更新训练进度
            update_training_progress(
                epoch + 1, epoches, avg_loss, val_loss, acc / 100, 'running',
                f"已完成第 {epoch + 1}/{epoches} 轮训练"
            )

            if acc > best_acc:
                best_acc = acc
                counter = 0  # 重置早停计数器

                torch.save(model.state_dict(), model_path)
                print(f'最佳模型已保存，准确率为: {best_acc:.4f}%')
            else:
                counter += 1  # 增加早停计数器
                print(f'早停计数器: {counter}/{patience}')

            # 执行学习率调度
            scheduler.step()

            # 检查早停条件
            if counter >= patience:
                print(f'早停机制触发，在第{epoch + 1}轮训练后停止')
                update_training_progress(
                    epoch + 1, epoches, avg_loss, val_loss, acc / 100, 'completed',
                    f"早停机制触发，在第{epoch + 1}轮训练后停止"
                )
                break

            # 最后保存所有训练日志，使用更详细的命名方式
            history_df = pd.DataFrame(history)

            log = os.path.join("log", args.model)
            os.makedirs(log, exist_ok=True)

            # 根据模型类型创建不同的日志文件名
            if args.model.lower() == "cnn":
                log_filename = os.path.join(log, f"{args.model.lower()}_dp{args.dropout}_wd{args.weight_decay}.csv")
            else:
                log_filename = os.path.join(log,
                                            f"{args.model.lower()}_dp{args.dropout}_hd{args.hidden_dim}_wd{args.weight_decay}.csv")
            history_df.to_csv(log_filename, index=False)

            print(f'训练日志已保存到: {log_filename}')
        except Exception as e:
            print(f"验证过程出错: {str(e)}")
            traceback.print_exc()
            update_training_progress(
                epoch + 1, epoches, avg_loss, None, None, 'failed', f"验证过程出错: {str(e)}"
            )
            continue

    # 训练完成后，更新最终状态
    update_training_progress(
        epoches, epoches, history["train_loss"][-1] if history["train_loss"] else None,
        history["val_loss"][-1] if history["val_loss"] else None,
        history["val_acc"][-1] / 100 if history["val_acc"] else None,
        'completed', "训练完成"
    )


def update_training_progress(current_epoch, total_epochs, train_loss, val_loss, val_acc, status, message=None):
    """更新训练进度文件"""
    progress_file = '/Users/joey/PycharmProjects/shebi/config/progress.json'
    try:
        # 确保文件存在
        if not os.path.exists(progress_file):
            with open(progress_file, 'w') as f:
                json.dump({
                    'status': 'initializing',
                    'current_epoch': 0,
                    'total_epochs': total_epochs,
                    'train_loss': 0,
                    'val_loss': 0,
                    'val_acc': 0,
                    'history': {
                        'train_loss': [],
                        'val_loss': [],
                        'train_acc': [],
                        'val_acc': []
                    }
                }, f)

        # 读取现有进度
        with open(progress_file, 'r') as f:
            progress = json.load(f)

        # 更新进度信息
        progress['current_epoch'] = current_epoch
        progress['total_epochs'] = total_epochs
        progress['status'] = status

        if train_loss is not None:
            progress['train_loss'] = float(train_loss)  # 确保数值类型
        if val_loss is not None:
            progress['val_loss'] = float(val_loss)  # 确保数值类型
        if val_acc is not None:
            progress['val_acc'] = float(val_acc)  # 确保数值类型

        if message:
            progress['message'] = message

        # 确保历史记录结构存在
        if 'history' not in progress:
            progress['history'] = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

        # 更新历史记录
        if train_loss is not None:
            progress['history']['train_loss'].append(float(train_loss))
        if val_loss is not None:
            progress['history']['val_loss'].append(float(val_loss))
        if val_acc is not None:
            progress['history']['val_acc'].append(float(val_acc))

        # 如果状态为失败，添加错误信息
        if status == 'failed' and message:
            progress['error'] = message
        elif status != 'failed' and 'error' in progress:
            progress.pop('error', None)  # 移除错误信息

        # 写入文件
        with open(progress_file, 'w') as f:
            json.dump(progress, f, indent=2)  # 使用缩进格式化JSON，便于调试

        print(f"进度已更新: 轮次 {current_epoch}/{total_epochs}, 状态: {status}")
    except Exception as e:
        print(f"更新训练进度时出错: {str(e)}")
        traceback.print_exc()


def load_params():
    """从 JSON 文件加载超参数"""
    try:
        params_file = '/Users/joey/PycharmProjects/shebi/config/params.json'
        with open(params_file, 'r') as f:
            params = json.load(f)
        print(f"加载的超参数: {params}")
        return params
    except Exception as e:
        print(f"加载参数文件出错: {str(e)}")
        traceback.print_exc()
        return {}


if __name__ == "__main__":
    try:
        # 配置日志目录
        os.makedirs("log", exist_ok=True)
        os.makedirs(Config.model_dir, exist_ok=True)  # 确保模型目录存在

        # 记录训练开始
        update_training_progress(0, 0, None, None, None, 'initializing', "初始化训练")

        # 加载超参数
        params = load_params()

        # 添加命令行参数解析
        parser = argparse.ArgumentParser(description='Text Classification Model Training')
        parser.add_argument('--model', type=str, default=params.get('model_type', 'cnn'),
                            choices=['bilstm_attention', 'bilstm', 'lstm_attention', 'lstm', 'cnn'],
                            help='选择使用的模型类型: BiLSTM_attention, BiLSTM, LSTM_attention, LSTM 或 TextCNN (默认: TextCNN)')
        parser.add_argument('--batch-size', type=int, default=params.get('batch_size', Config.batch_size),
                            help='批量大小 (默认: 64)')
        parser.add_argument('--dropout', type=float, default=params.get('dropout', Config.dropout),
                            help='选择 dropout (默认: Config.dropout)')
        parser.add_argument('--hidden-dim', type=int, default=params.get('hidden_dim', Config.hidden_dim),
                            help='隐藏层维度 (默认: Config.hidden_dim)')
        parser.add_argument('--embedding-dim', type=int, default=params.get('embedding_dim', Config.embedding_dim),
                            help='嵌入层维度 (默认: Config.embedding_dim)')
        parser.add_argument('--num-layers', type=int, default=params.get('num_layers', Config.num_layers),
                            help='LSTM层数 (默认: Config.num_layers)')
        parser.add_argument('--num-filters', type=int, default=params.get('num_filters', Config.num_filters),
                            help='卷积核数量 (默认: Config.num_filters)')
        parser.add_argument('--patience', type=int, default=params.get('patience', Config.patience),
                            help='早停耐心值 (默认: 10)')
        parser.add_argument('--learning-rate', type=float, default=params.get('learning_rate', Config.lr),
                            help='学习率 (默认: Config.lr)')
        parser.add_argument('--weight-decay', type=float, default=params.get('weight_decay', 1e-4),
                            help='权重衰减 (默认: 1e-4)')
        parser.add_argument('--epochs', type=int, default=params.get('epochs', Config.n_epoch),
                            help='训练迭代周期 (默认: 10)')
        parser.add_argument('--seed', type=int, default=42, help='随机种子，用于实验的可重复性 (默认: 42)')
        parser.add_argument('--pad-size', type=int, default=Config.pad_size, help='填充大小 (默认: 32)')

        args = parser.parse_args()

        # 确保lr参数被正确设置
        Config.lr = args.learning_rate
        Config.n_epoch = args.epochs

        # 更新训练状态
        update_training_progress(0, args.epochs, None, None, None, 'preparing', "加载数据和预处理")

        # 设置随机种子
        set_seed(args.seed)

        # 打印所有命令行参数的值
        print("\n" + "=" * 50)
        print("运行配置参数:")
        print("=" * 50)
        for arg in vars(args):
            print(f"{arg}: {getattr(args, arg)}")
        print("=" * 50 + "\n")

        # 主函数：预览数据、预处理、模型构建、训练和保存模型
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {device}")

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
        train_dataloader = DataLoader(train_loader, batch_size=args.batch_size, shuffle=True,
                                      num_workers=0)  # 注意：num_workers设置为0时速度较快

        val_loader = Data_set(val_array, val_label)
        val_dataloader = DataLoader(val_loader, batch_size=args.batch_size, shuffle=True, num_workers=0)

        test_loader = Data_set(test_array, test_label)
        test_dataloader = DataLoader(test_loader, batch_size=args.batch_size, shuffle=True, num_workers=0)

        # 使用导入的创建模型函数，传入args参数
        model = create_model(
            args.model,
            w2vec,
            device=device,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout
        )
        print(f'使用 {args.model.upper()} 模型训练')

        # 保存模型（根据模型名字保存，并添加缓存）
        model_filename = f"{args.model}_model_best.pkl"
        model_path = os.path.join(Config.model_dir, model_filename)
        print(f"模型将保存到: {model_path}")

        # 训练模型
        train(train_dataloader, model=model, device=device, epoches=args.epochs, lr=args.learning_rate,
              patience=args.patience)

        # 保存最终模型
        try:
            torch.save(model.state_dict(), model_path)
            print(f"模型已保存为: {model_path}")
        except Exception as e:
            print(f"保存模型出错: {str(e)}")
            traceback.print_exc()
            update_training_progress(
                args.epochs, args.epochs, None, None, None, 'failed', f"保存模型出错: {str(e)}"
            )

    except Exception as e:
        print(f"训练过程出错: {str(e)}")
        traceback.print_exc()
        update_training_progress(0, 0, None, None, None, 'failed', f"训练过程出错: {str(e)}")
        sys.exit(1)
