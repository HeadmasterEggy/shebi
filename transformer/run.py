# coding: UTF-8
import time
import torch
import numpy as np
import os
from utils import build_dataset, build_iterator, get_time_dif
from train_eval import train, init_network
from importlib import import_module
import argparse

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
args = parser.parse_args()


if __name__ == '__main__':
    # 获取当前文件的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 获取项目根目录
    project_root = os.path.dirname(current_dir)
    
    # 设置路径
    dataset = os.path.join(project_root, 'data')  # 数据集
    embedding_path = os.path.join(project_root, 'word2vec', 'wiki_word2vec_50.bin')  # 预训练词向量
    
    # 打印调试信息
    print(f"当前目录: {current_dir}")
    print(f"项目根目录: {project_root}")
    print(f"数据集路径: {dataset}")
    print(f"词向量路径: {embedding_path}")
    print(f"数据集路径存在: {os.path.exists(dataset)}")
    print(f"词向量路径存在: {os.path.exists(embedding_path)}")

    model_name = args.model

    # 导入模型模块
    try:
        x = import_module('models.' + model_name)
        config = x.Config(dataset, embedding_path)
    except Exception as e:
        print(f"导入模型时出错: {e}")
        import sys
        sys.exit(1)
        
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    vocab, train_data, dev_data, test_data = build_dataset(config, args.word)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    config.n_vocab = len(vocab)
    model = x.Model(config).to(config.device)
    if model_name != 'Transformer':
        init_network(model)
    print(model)  # 打印模型结构，而不是model.parameters
    train(config, model, train_iter, dev_iter, test_iter)