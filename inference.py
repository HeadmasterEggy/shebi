# -*- coding: utf-8 -*-
"""推理资源的进程内单例。

改造前 /api/analyze 每收到一个请求都会重头做一遍：

    1. build_word2id()  —— 遍历 train.txt(50674 行) + val.txt(6334 行) 重建词表并回写磁盘
    2. prepare_data()   —— 把训练/验证/测试三个集合全部转成索引矩阵
    3. build_word2vec() —— 用 gensim 载入 12MB 二进制词向量，再拼一个 60723x50 的矩阵
    4. torch.load()     —— 重新反序列化模型权重

实测单请求 17.4s，其中 99% 与用户输入无关。这些资源在进程生命周期内是不变的，
所以这里把它们收成一个带锁的懒加载单例，启动时预热一次，之后每个请求只做前向。
"""

import logging
import os
import threading

import numpy as np
import torch

from config import Config
from data_Process import build_word2id, build_word2vec
from utils import initialize_model

logger = logging.getLogger(__name__)


def load_word2id(path=None):
    """优先读取已生成的 word2id.txt；文件不存在时才回退到全量重建。

    build_word2id() 会遍历整个训练集并回写磁盘，只适合离线执行一次。
    """
    path = path or Config.word2id_path
    if not os.path.exists(path):
        logger.warning("词表文件不存在，正在重建：%s", path)
        return build_word2id(path)

    word2id = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 2:
                word2id[parts[0]] = int(parts[1])
    logger.info("词表已加载：%d 个词（%s）", len(word2id), path)
    return word2id


class InferenceEngine:
    """线程安全的推理资源容器。"""

    def __init__(self, device=None):
        self.device = device or torch.device("cpu")
        # 必须是可重入锁：w2vec 的懒加载内部会再取 word2id，
        # get_model 内部又会再取 w2vec，普通 Lock 在这里会自锁死。
        self._lock = threading.RLock()
        self._word2id = None
        self._w2vec = None
        self._models = {}

    # ---------- 词表 ----------
    @property
    def word2id(self):
        if self._word2id is None:
            with self._lock:
                if self._word2id is None:
                    self._word2id = load_word2id()
        return self._word2id

    # ---------- 词向量 ----------
    @property
    def w2vec(self):
        if self._w2vec is None:
            with self._lock:
                if self._w2vec is None:
                    logger.info("加载预训练词向量：%s", Config.pre_word2vec_path)
                    mat = build_word2vec(Config.pre_word2vec_path, self.word2id, None)
                    self._w2vec = torch.from_numpy(np.asarray(mat)).float()
                    logger.info("词向量矩阵：%s", tuple(self._w2vec.shape))
        return self._w2vec

    # ---------- 模型 ----------
    def get_model(self, model_type=None):
        model_type = (model_type or Config.default_model).lower()
        if model_type not in Config.model_choices:
            raise ValueError(f"不支持的模型类型: {model_type}")

        model = self._models.get(model_type)
        if model is None:
            with self._lock:
                model = self._models.get(model_type)
                if model is None:
                    logger.info("首次加载模型：%s", model_type)
                    model = initialize_model(model_type, self.w2vec, self.device)
                    model.eval()
                    self._models[model_type] = model
        return model

    def invalidate_model(self, model_type=None):
        """训练产出新权重后调用，让下一次请求重新加载。"""
        with self._lock:
            if model_type:
                self._models.pop(model_type.lower(), None)
            else:
                self._models.clear()

    def warmup(self, model_type=None):
        """启动时预热，把首个请求的冷启动成本挪到进程启动阶段。"""
        self.get_model(model_type or Config.default_model)
        logger.info("推理引擎预热完成")


engine = InferenceEngine()
