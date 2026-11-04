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


def pretrained_weight_size(w2vec):
    """返回预训练词向量矩阵的词表大小。"""
    return int(w2vec.shape[0])



def create_model(model_type, w2vec, device=None, embedding_dim=None, hidden_dim=None,
                 num_layers=None, dropout=None, num_filters=None, filter_sizes=None):
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
    model_type = model_type.lower() if model_type else Config.default_model

    # 必须拷一份再交给模型。nn.Embedding.from_pretrained 不复制张量，而是直接
    # 拿它当 weight，于是模型的 embedding 与传进来的 w2vec 共享存储；随后
    # initialize_model 里的 load_state_dict 会原地写入，把调用方的预训练词向量
    # 一并改掉（实测最大绝对差 1.61）。后果有二：
    #   1. 接着初始化第二个模型时，用的是上一个模型微调过的 embedding，而不是预训练值
    #   2. 依赖 engine.w2vec 的下游（向量索引、句向量编码）会在"载入模型前后"
    #      落到两个不同的向量空间里
    w2vec = w2vec.clone() if hasattr(w2vec, "clone") else w2vec

    # 词表大小以实际词向量矩阵为准，而不是 Config 里写死的常量。
    # Config.vocab_size 曾长期停留在 54848，而 build_word2id 依据当前数据集
    # 实际产出 60723 个词；两者不一致时 padding_idx 会落到一个真实词上。
    vocab_size = pretrained_weight_size(w2vec)

    # 如果参数为None，则使用Config默认值
    embedding_dim = embedding_dim if embedding_dim is not None else Config.embedding_dim
    hidden_dim = hidden_dim if hidden_dim is not None else Config.hidden_dim
    num_layers = num_layers if num_layers is not None else Config.num_layers
    dropout = dropout if dropout is not None else Config.dropout
    num_filters = num_filters if num_filters is not None else Config.num_filters
    filter_sizes = tuple(filter_sizes) if filter_sizes else tuple(Config.filter_sizes)

    if model_type == 'bilstm_attention':
        model = LSTM_attention(
            vocab_size, embedding_dim, w2vec, Config.update_w2v,
            hidden_dim, num_layers, dropout,
            Config.n_class, Config.bidirectional_1
        )
    elif model_type == 'bilstm':
        model = LSTMModel(
            vocab_size, embedding_dim, w2vec, Config.update_w2v,
            hidden_dim, num_layers, dropout,
            Config.n_class, Config.bidirectional_1
        )
    elif model_type == 'lstm_attention':
        model = LSTM_attention(
            vocab_size, embedding_dim, w2vec, Config.update_w2v,
            hidden_dim, num_layers, dropout,
            Config.n_class, Config.bidirectional_2
        )
    elif model_type == 'lstm':
        model = LSTMModel(
            vocab_size, embedding_dim, w2vec, Config.update_w2v,
            hidden_dim, num_layers, dropout,
            Config.n_class, Config.bidirectional_2
        )
    else:  # 默认使用CNN模型
        model = TextCNN(
            dropout,
            vocab_size,
            Config.pad_size,
            filter_sizes,
            num_filters,
            w2vec,
            embedding_dim,
            Config.n_class
        )

    # 如果指定了设备，则将模型移动到该设备
    if device is not None:
        model = model.to(device)

    return model


class ModelNotAvailable(FileNotFoundError):
    """权重文件缺失或无法加载（例如只拉到了 Git LFS 指针文件）。"""


# 决定权重张量形状的结构超参。训练时随权重一起落盘，加载时据此重建同样结构的模型。
#
# 原来加载一律按 Config 默认值建模型：在网页上把 LSTM 的 hidden_dim 改成 64 训练，
# 产出的权重就再也装不进那个按 128 建出来的模型（load_state_dict 报 size mismatch）。
ARCH_KEYS = ("hidden_dim", "num_layers", "num_filters", "filter_sizes", "embedding_dim")


def arch_path(weight_path):
    """权重文件对应的结构描述文件：cnn_model_best.pkl -> cnn_model_best.arch.json"""
    root, _ = os.path.splitext(weight_path)
    return root + ".arch.json"


def save_checkpoint(state_dict, weight_path, arch):
    """原子地保存权重与结构描述。

    先写结构、后写权重：推理引擎按权重文件的指纹判断是否重载，权重落盘的那一刻
    结构描述已经是新的。两者都先写临时文件再 os.replace，Web 进程不会读到写了一半的文件。
    """
    import json

    arch = {k: (list(v) if isinstance(v, tuple) else v) for k, v in arch.items() if k in ARCH_KEYS}
    tmp = arch_path(weight_path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(arch, f, ensure_ascii=False, indent=2)
    os.replace(tmp, arch_path(weight_path))

    tmp = weight_path + ".tmp"
    torch.save(state_dict, tmp)
    os.replace(tmp, weight_path)


def load_arch(weight_path):
    import json

    try:
        with open(arch_path(weight_path), encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}  # 旧权重没有结构描述：按 Config 默认值建模
    return {k: v for k, v in data.items() if k in ARCH_KEYS}


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

    model_filename = f"{model_type.lower()}_model_best.pkl"
    best_model_path = os.path.join(Config.model_dir, model_filename)

    # 按训练时记录的结构建模型，没有记录就用 Config 默认值
    arch = load_arch(best_model_path)
    model = create_model(model_type, w2vec, device, **arch)


    logging.info(f"模型文件路径: {best_model_path}")

    # 加载最优模型。
    #
    # 原实现在文件缺失或加载失败时只打一条警告，然后返回一个随机初始化的模型——
    # 线上照常给出看起来很像样的置信度，实际是随机数。刚 clone 下来、Git LFS 只拉到
    # 133 字节指针文件时，五个模型全部是这种状态。现在宁可报错也不给假结果。
    if not os.path.exists(best_model_path):
        raise ModelNotAvailable(f"{model_type} 模型尚未训练（找不到 {best_model_path}）")
    # weights_only=True：只反序列化张量与基础容器。原来用 weights_only=False，
    # 等于对权重文件做任意 pickle 反序列化——换掉一个 .pkl 就能在服务进程里执行代码。
    # main.py 一直只保存 state_dict，不需要完整 pickle。
    try:
        state_dict = torch.load(best_model_path, map_location=device, weights_only=True)
    except Exception as e:
        raise ModelNotAvailable(f"{model_type} 模型权重无法加载（{e}），"
                                f"可能只拉到了 Git LFS 指针文件，请重新训练或 git lfs pull") from e
    if not isinstance(state_dict, dict):
        raise ModelNotAvailable(f"{best_model_path} 不是 state_dict，请用 main.py 重新训练")
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        # 结构对不上（size mismatch / missing keys）。原来这个异常会原样冒出去，
        # 接口只能返回一个笼统的 500。
        raise ModelNotAvailable(
            f"{model_type} 权重与模型结构不匹配（结构参数 {arch or 'Config 默认值'}），"
            f"请重新训练：{str(e).splitlines()[0]}") from e
    logging.info(f"成功加载模型: {best_model_path} 到 {device} 设备")

    model.to(device)
    model.eval()  # 设置为评估模式
    return model
