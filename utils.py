"""
模型工具模块 - 包含模型创建和初始化的共享函数
"""
import logging
import os

import torch

from cnn_model import TextCNN
from config import Config
from lstm_model import LSTM_attention, LSTMModel

logger = logging.getLogger(__name__)


def create_model(model_type, w2vec, device=None, embedding_dim=None, hidden_dim=None,
                 num_layers=None, dropout=None):
    """
    根据模型类型创建对应的模型实例

    参数:
        model_type: 模型类型
        w2vec: 词向量
        device: 可选，运行设备
        embedding_dim: 可选，词嵌入维度，如果None则使用Config默认值
        hidden_dim: 可选，隐藏层维度，如果None则使用Config默认值
        num_layers: 可选，LSTM层数，如果None则使用Config默认值
        dropout: 可选，dropout比率，如果None则使用Config默认值

    返回:
        对应类型的模型实例
    """
    model_type = model_type.lower() if model_type else Config.model_name.lower()

    # 如果参数为None，则使用Config默认值
    embedding_dim = embedding_dim if embedding_dim is not None else Config.embedding_dim
    hidden_dim = hidden_dim if hidden_dim is not None else Config.hidden_dim
    num_layers = num_layers if num_layers is not None else Config.num_layers
    dropout = dropout if dropout is not None else Config.dropout

    if model_type == 'bilstm_attention':
        model = LSTM_attention(
            Config.vocab_size, embedding_dim, w2vec, Config.update_w2v,
            hidden_dim, num_layers, dropout,
            Config.n_class, Config.bidirectional_1
        )
    elif model_type == 'bilstm':
        model = LSTMModel(
            Config.vocab_size, embedding_dim, w2vec, Config.update_w2v,
            hidden_dim, num_layers, dropout,
            Config.n_class, Config.bidirectional_1
        )
    elif model_type == 'lstm_attention':
        model = LSTM_attention(
            Config.vocab_size, embedding_dim, w2vec, Config.update_w2v,
            hidden_dim, num_layers, dropout,
            Config.n_class, Config.bidirectional_2
        )
    elif model_type == 'lstm':
        model = LSTMModel(
            Config.vocab_size, embedding_dim, w2vec, Config.update_w2v,
            hidden_dim, num_layers, dropout,
            Config.n_class, Config.bidirectional_2
        )
    else:  # 默认使用CNN模型
        model = TextCNN(
            dropout,
            Config.vocab_size,
            Config.pad_size,
            Config.filter_sizes,
            Config.num_filters,
            w2vec,
            embedding_dim,
            Config.n_class
        )

    # 如果指定了设备，则将模型移动到该设备
    if device is not None:
        model = model.to(device)

    return model


def initialize_model(model_type, w2vec, device=torch.device('cpu')):
    """
    初始化模型并加载最优模型。

    参数:
        model_type: 模型类型
        w2vec: 词向量
        device: 运行设备，默认为CPU

    返回:
        加载了权重的模型实例
    """
    logging.info(f"使用 {model_type.upper()} 模型")

    # 创建模型实例
    model = create_model(model_type, w2vec, device)

    # 构建模型文件路径
    model_filename = f"{model_type.lower()}_model_best.pkl"
    best_model_path = os.path.join(Config.model_dir, model_filename)

    logging.info(f"模型文件路径: {best_model_path}")

    # 加载最优模型
    if os.path.exists(best_model_path):
        try:
            loaded_model = torch.load(best_model_path, map_location=device, weights_only=False)
            if isinstance(loaded_model, dict):
                model.load_state_dict(loaded_model)
            else:
                model = loaded_model
            logging.info(f"成功加载模型: {best_model_path} 到 {device} 设备")
        except Exception as e:
            logging.error(f"加载模型失败: {e}")
            logging.warning(f"使用未初始化的 {model_type} 模型")
    else:
        logging.warning(f"找不到 {model_type.upper()} 最佳模型，请确保模型文件存在: {best_model_path}")

    model.to(device)
    model.eval()  # 设置为评估模式
    return model
