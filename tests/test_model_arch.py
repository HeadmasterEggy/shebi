# -*- coding: utf-8 -*-
"""结构超参随权重一起落盘，推理端按它重建模型。

原来推理一律按 Config 默认值建模型：网页上把 LSTM 的 hidden_dim 改成 64 训练出来的
权重，load_state_dict 时 size mismatch，再也加载不了；CNN 的卷积核数量 / 尺寸与优化器
选项则在训练端就被直接忽略。
"""

import os

import pytest
import torch

from config import Config
from utils import ModelNotAvailable, arch_path, create_model, initialize_model, load_arch, save_checkpoint


@pytest.fixture
def model_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "model_dir", str(tmp_path))
    return tmp_path


@pytest.fixture
def w2v():
    torch.manual_seed(0)
    return torch.randn(50, Config.embedding_dim)


@pytest.mark.parametrize("model_type,arch", [
    ("lstm", {"hidden_dim": 32, "num_layers": 1}),
    ("bilstm_attention", {"hidden_dim": 16, "num_layers": 3}),
    ("cnn", {"num_filters": 8, "filter_sizes": [2, 5]}),
])
def test_non_default_architecture_roundtrips(model_dir, w2v, model_type, arch):
    trained = create_model(model_type, w2v, **arch)
    path = os.path.join(str(model_dir), f"{model_type}_model_best.pkl")
    save_checkpoint(trained.state_dict(), path, arch)

    loaded = initialize_model(model_type, w2v)
    x = torch.randint(1, 50, (3, Config.max_sen_len))
    trained.eval()
    assert torch.allclose(trained(x), loaded(x), atol=1e-6), "加载回来的模型与训练时的不一致"


def test_checkpoint_writes_are_atomic_and_arch_is_filtered(model_dir, w2v):
    path = os.path.join(str(model_dir), "cnn_model_best.pkl")
    save_checkpoint(create_model("cnn", w2v).state_dict(), path,
                    {"num_filters": 128, "filter_sizes": (2, 3, 4), "dropout": 0.5, "junk": 1})
    assert sorted(os.listdir(model_dir)) == ["cnn_model_best.arch.json", "cnn_model_best.pkl"]
    assert load_arch(path) == {"num_filters": 128, "filter_sizes": [2, 3, 4]}


def test_mismatched_weights_raise_model_not_available(model_dir, w2v):
    """没有结构描述的旧权重若与默认结构不符，要明确报错，而不是冒出一个裸 RuntimeError（接口 500）。"""
    path = os.path.join(str(model_dir), "lstm_model_best.pkl")
    torch.save(create_model("lstm", w2v, hidden_dim=8, num_layers=1).state_dict(), path)
    assert not os.path.exists(arch_path(path))
    with pytest.raises(ModelNotAvailable, match="结构不匹配"):
        initialize_model("lstm", w2v)


def test_legacy_weights_without_arch_file_still_load(model_dir, w2v):
    path = os.path.join(str(model_dir), "cnn_model_best.pkl")
    torch.save(create_model("cnn", w2v).state_dict(), path)
    initialize_model("cnn", w2v)


@pytest.mark.parametrize("name,cls", [("adam", torch.optim.Adam), ("sgd", torch.optim.SGD),
                                      ("rmsprop", torch.optim.RMSprop)])
def test_optimizer_choice_is_honoured(name, cls):
    from main import build_optimizer
    opt = build_optimizer(name, [torch.nn.Parameter(torch.zeros(1))], 1e-3, 0.0)
    assert isinstance(opt, cls)


def test_unknown_optimizer_rejected():
    from main import build_optimizer
    with pytest.raises(ValueError):
        build_optimizer("lbfgs", [torch.nn.Parameter(torch.zeros(1))], 1e-3, 0.0)


def test_filter_sizes_parsing():
    from main import parse_filter_sizes
    assert parse_filter_sizes("4, 2,3,2") == (2, 3, 4)
    assert parse_filter_sizes([5]) == (5,)
    with pytest.raises(ValueError):
        parse_filter_sizes("")
