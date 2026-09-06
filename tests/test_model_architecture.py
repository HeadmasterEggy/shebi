# -*- coding: utf-8 -*-
"""锁死那个让整个毕设实验结论反转的缺陷。

原实现在 TextCNN 里用了 nn.BatchNorm1d。BatchNorm 在 train() 下用当前 batch 统计量、
在 eval() 下用 running stats，而嵌入层参与训练导致特征分布持续漂移，running stats
始终追不上 —— 最终 eval() 下模型把验证集 6334 条样本全部判成同一类（acc 49.87%）。

线上 /api/analyze 永远走 eval()，于是任何一条差评都被判成"积极"。
test_train_and_eval_modes_agree 与 test_no_batchnorm_layers 各自都能独立抓到这个回归
（已验证：换回 BatchNorm 后二者立即失败）。
"""

import torch

from cnn_model import TextCNN
from config import Config


def _build(vocab=500, dim=Config.embedding_dim):
    torch.manual_seed(0)
    weight = torch.randn(vocab, dim)
    return TextCNN(
        dropout=Config.dropout, vocab_size=vocab, pad_size=Config.pad_size,
        filter_sizes=Config.filter_sizes, num_filters=Config.num_filters,
        pretrained_weight=weight, embedding_dim=dim, n_class=Config.n_class,
    )


def test_train_and_eval_modes_agree():
    """train() 与 eval() 对同一批输入必须给出相同 logits（dropout 关掉后）。

    BatchNorm 做不到这一点，LayerNorm 可以。这是本次缺陷的直接判据。
    """
    model = _build()
    x = torch.randint(1, 500, (16, Config.max_sen_len))

    model.eval()
    with torch.no_grad():
        out_eval = model(x)

    # 只把归一化层切回 train，dropout 保持关闭，隔离出归一化层的行为差异
    model.norm.train()
    with torch.no_grad():
        out_train_norm = model(x)

    assert torch.allclose(out_eval, out_train_norm, atol=1e-6), (
        "归一化层在 train/eval 下行为不一致——BatchNorm 回归了"
    )


def test_eval_output_independent_of_batch_composition():
    """单条样本单独推理，与它混在一个 batch 里推理，结果必须一致。

    这条不是用来抓原始缺陷的（eval 模式的 BatchNorm 走 running stats，同样满足这条），
    而是用来挡住一种常见的"错误修法"：把 BatchNorm 改成 track_running_stats=False。
    那样 eval 也会用 batch 统计量，线上单条请求（batch=1）的结果就会随同批样本漂移。
    """
    model = _build()
    model.eval()
    torch.manual_seed(1)
    batch = torch.randint(1, 500, (8, Config.max_sen_len))

    with torch.no_grad():
        grouped = model(batch)
        alone = torch.cat([model(batch[i:i + 1]) for i in range(batch.shape[0])])

    assert torch.allclose(grouped, alone, atol=1e-5), (
        "推理结果依赖 batch 内其他样本——归一化层用了 batch 统计量"
    )


def test_no_batchnorm_layers():
    """结构层面禁止再出现 BatchNorm。"""
    offenders = [name for name, m in _build().named_modules()
                 if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d))]
    assert not offenders, f"TextCNN 中不应出现 BatchNorm: {offenders}"


def test_padding_idx_points_at_pad_token():
    """padding_idx 必须指向 _PAD_ 的真实索引 0。

    旧实现用 vocab_size - 1，等于把词表末尾的一个真实词当成了填充符，
    而且 Config.vocab_size(54848) 与真实词表(60723)本就对不上。
    """
    model = _build()
    assert model.padding_idx == Config.pad_idx == 0
    assert model.embedding.padding_idx == 0


def test_vocab_size_follows_embedding_matrix():
    """模型词表大小以词向量矩阵为准，而不是 Config 里的常量。"""
    from utils import create_model
    w2vec = torch.randn(1234, Config.embedding_dim)
    model = create_model("cnn", w2vec)
    assert model.embedding.num_embeddings == 1234
