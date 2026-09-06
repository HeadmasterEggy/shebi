# -*- coding: utf-8 -*-
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("SHEBI_ALLOW_DEV_SECRET", "1")

from config import Config  # noqa: E402


def _has_weights(model_type):
    path = os.path.join(Config.model_dir, f"{model_type}_model_best.pkl")
    # 小于 1KB 说明是没拉下来的 Git LFS 指针文件
    return os.path.exists(path) and os.path.getsize(path) > 1024


needs_weights = pytest.mark.skipif(
    not _has_weights(Config.default_model),
    reason=f"缺少 {Config.default_model} 权重，先运行 main.py 训练",
)

needs_w2v = pytest.mark.skipif(
    not (os.path.exists(Config.pre_word2vec_path)
         and os.path.getsize(Config.pre_word2vec_path) > 1024),
    reason="缺少预训练词向量（Git LFS 未拉取），可运行 scripts/rebuild_pretrained_w2v.py 重建",
)
