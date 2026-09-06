# -*- coding: utf-8 -*-
"""推理单例：缓存有效、不死锁、eval 下不坍缩。"""

import time

import numpy as np
import pytest
import torch

from config import Config
from tests.conftest import needs_weights, needs_w2v


@needs_w2v
def test_word2id_is_loaded_not_rebuilt():
    """词表应从 word2id.txt 直接读取，而不是遍历 5 万行训练集重建。"""
    from inference import InferenceEngine
    eng = InferenceEngine()
    t0 = time.perf_counter()
    w = eng.word2id
    elapsed = time.perf_counter() - t0
    assert len(w) > 1000
    assert w["_PAD_"] == 0
    assert elapsed < 3.0, f"加载词表耗时 {elapsed:.1f}s，疑似退化成全量重建"


@needs_w2v
def test_resources_are_cached_across_calls():
    from inference import InferenceEngine
    eng = InferenceEngine()
    assert eng.word2id is eng.word2id
    assert eng.w2vec is eng.w2vec


@needs_w2v
@needs_weights
def test_get_model_is_cached_and_does_not_deadlock():
    """get_model 内部会取 w2vec，w2vec 内部又会取 word2id。

    这些懒加载互相嵌套，若用不可重入的 threading.Lock 会直接自锁死
    （曾经就是这样，evaluate.py 卡住不动）。
    """
    from inference import InferenceEngine
    eng = InferenceEngine()
    m1 = eng.get_model(Config.default_model)   # 冷启动，三层嵌套一次走完
    m2 = eng.get_model(Config.default_model)
    assert m1 is m2


@needs_w2v
@needs_weights
def test_model_is_returned_in_eval_mode():
    from inference import engine
    assert engine.get_model(Config.default_model).training is False


@needs_w2v
@needs_weights
def test_rejects_unknown_model_type():
    from inference import engine
    with pytest.raises(ValueError):
        engine.get_model("transformer_xxl")


@needs_w2v
@needs_weights
def test_predictions_do_not_collapse_to_one_class():
    """最直接的回归判据：eval 模式下，测试集预测不能全部落在同一个类。

    修复前 TextCNN 把 6334 条验证样本 100% 判成正类（acc 49.87%，恰好等于正类占比）。
    """
    from data_Process import text_to_array
    from inference import engine

    word2id = engine.word2id
    arr, labels = text_to_array(word2id, Config.max_sen_len, Config.test_path)
    sample = torch.tensor(arr[:1024], dtype=torch.long)

    model = engine.get_model(Config.default_model)
    with torch.no_grad():
        preds = model(sample).argmax(1).numpy()

    dist = np.bincount(preds, minlength=Config.n_class)
    assert dist.min() > 0, f"模型坍缩到单一类别，预测分布={dist.tolist()}"

    acc = (preds == np.array(labels[:1024])).mean()
    assert acc > 0.75, f"测试集准确率仅 {acc:.2%}，明显低于预期"
