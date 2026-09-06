# -*- coding: utf-8 -*-
"""工具实现。

设计上有三条硬规则：

1. **批量优先**。classify_sentiment 收一个 list 而不是单条。让模型一次传 200 条，
   而不是循环调 200 次——这是 agent 工程里最常见、也最贵的性能坑。
2. **可溯源**。凡是返回评论的工具都带 review_id，Critic 才有东西可校验。
3. **有副作用的工具显式标记**。爬虫要发外部请求、训练要占算力，
   requires_confirmation=True，由循环决定是否需要人工放行。
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import torch
from pydantic import BaseModel, Field, field_validator

from config import Config
from agent import store
from agent.registry import ToolError, registry

MAX_BATCH = 512


# --------------------------------------------------------------------------
# classify_sentiment —— 把毕设自训练的模型包成工具
# --------------------------------------------------------------------------
class ClassifyArgs(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=MAX_BATCH,
                             description="待分类的文本列表。一次尽量多传，不要逐条调用。")
    model: Optional[Literal["cnn", "lstm", "bilstm", "lstm_attention", "bilstm_attention"]] = Field(
        None, description="使用哪个本地模型，留空则用默认模型")

    @field_validator("texts")
    @classmethod
    def _strip(cls, v):
        out = [t.strip() for t in v if t and t.strip()]
        if not out:
            raise ValueError("texts 不能全是空字符串")
        return out


@registry.register(
    name="classify_sentiment",
    description=(
        "用本地自训练的中文情感分类模型批量判断文本情感（积极/消极），返回每条的标签与置信度。"
        "推理成本几乎为零、单条约 0.5ms，凡是只需要极性判断的场景都应优先用它，不要用大模型去数星星。"
    ),
    args_model=ClassifyArgs,
    tags=["local-model"],
)
def classify_sentiment(texts: List[str], model: Optional[str] = None) -> Dict[str, Any]:
    from data_Process import clean_text, tokenize
    from inference import engine

    model_type = (model or Config.default_model).lower()
    net = engine.get_model(model_type)
    word2id = engine.word2id

    stopwords = _stopwords()
    seq_len = Config.max_sen_len

    arrays = []
    for t in texts:
        tokens = tokenize(clean_text(t), stopwords).split()
        ids = [word2id.get(w, Config.pad_idx) for w in tokens]
        if len(ids) < seq_len:
            ids = [Config.pad_idx] * (seq_len - len(ids)) + ids
        else:
            ids = ids[:seq_len]
        arrays.append(ids)

    with torch.no_grad():
        logits = net(torch.tensor(np.array(arrays), dtype=torch.long))
        probs = torch.softmax(logits, dim=1).numpy()

    results = []
    for text, p in zip(texts, probs):
        idx = int(p.argmax())
        results.append({
            "text": text,
            "sentiment": "积极" if idx == 1 else "消极",
            "confidence": round(float(p[idx]) * 100, 2),
            "probabilities": {"negative": round(float(p[0]) * 100, 2),
                              "positive": round(float(p[1]) * 100, 2)},
        })

    n_pos = sum(1 for r in results if r["sentiment"] == "积极")
    return {
        "model": model_type,
        "count": len(results),
        "positive": n_pos,
        "negative": len(results) - n_pos,
        "results": results,
    }


_STOPWORDS: Optional[List[str]] = None


def _stopwords() -> List[str]:
    global _STOPWORDS
    if _STOPWORDS is None:
        with open(Config.stopword_path, encoding="utf-8") as f:
            _STOPWORDS = [line.strip() for line in f]
    return _STOPWORDS


# --------------------------------------------------------------------------
# search_reviews —— 检索，结论溯源的基础
# --------------------------------------------------------------------------
class SearchArgs(BaseModel):
    query: str = Field(..., min_length=1, description="检索词，例如「物流慢」「屏幕划痕」")
    k: int = Field(5, ge=1, le=50, description="返回条数")
    product_id: Optional[str] = Field(None, description="限定商品，留空则全库检索")


@registry.register(
    name="search_reviews",
    description=("按关键词检索评论库，返回带 review_id 的原文。"
                 "任何写进报告的结论都必须能引用到这里返回的 review_id。"),
    args_model=SearchArgs,
    tags=["retrieval"],
)
def search_reviews(query: str, k: int = 5, product_id: Optional[str] = None) -> Dict[str, Any]:
    hits = store.search(query, k=k, product_id=product_id)
    return {
        "query": query,
        "count": len(hits),
        "reviews": [{"review_id": h["id"], "text": h["text"],
                     "relevance": h["relevance"], "source": h["source"]} for h in hits],
    }


# --------------------------------------------------------------------------
# get_reviews —— 按 id 取原文，供 Critic 校验引用
# --------------------------------------------------------------------------
class GetReviewsArgs(BaseModel):
    review_ids: List[int] = Field(..., min_length=1, max_length=100,
                                  description="要取回的评论 id 列表")


@registry.register(
    name="get_reviews",
    description="按 review_id 取回评论原文。用于核实某条结论是否真的有原文支撑。",
    args_model=GetReviewsArgs,
    tags=["retrieval"],
)
def get_reviews(review_ids: List[int]) -> Dict[str, Any]:
    found, missing = [], []
    for rid in review_ids:
        row = store.get(rid)
        if row:
            found.append({"review_id": row["id"], "text": row["text"],
                          "source": row["source"], "score": row["score"]})
        else:
            missing.append(rid)
    return {"found": found, "missing": missing}


# --------------------------------------------------------------------------
# aggregate_reviews —— 统计
# --------------------------------------------------------------------------
class AggregateArgs(BaseModel):
    product_id: Optional[str] = Field(None, description="限定商品，留空则全库")
    sample_limit: int = Field(200, ge=1, le=MAX_BATCH,
                              description="参与情感统计的评论条数上限")
    model: Optional[Literal["cnn", "lstm", "bilstm", "lstm_attention", "bilstm_attention"]] = None


@registry.register(
    name="aggregate_reviews",
    description="对评论库做整体统计：总量、情感分布、正负比例。用于回答「口碑怎么样」这类总体性问题。",
    args_model=AggregateArgs,
    tags=["analysis"],
)
def aggregate_reviews(product_id: Optional[str] = None, sample_limit: int = 200,
                      model: Optional[str] = None) -> Dict[str, Any]:
    rows = store.fetch(limit=sample_limit, product_id=product_id)
    if not rows:
        raise ToolError("评论库里没有匹配的数据，先用 scrape_reviews 采集或检查 product_id", retryable=False)

    out = classify_sentiment([r["text"] for r in rows], model=model)
    total = out["count"]
    return {
        "product_id": product_id,
        "analyzed": total,
        "total_in_store": store.count(),
        "positive": out["positive"],
        "negative": out["negative"],
        "positive_rate": round(out["positive"] / total * 100, 2) if total else 0.0,
        "model": out["model"],
    }


# --------------------------------------------------------------------------
# scrape_reviews —— 有副作用，需人工确认
# --------------------------------------------------------------------------
class ScrapeArgs(BaseModel):
    product_url: str = Field(..., description="京东商品页链接")
    pages: int = Field(1, ge=1, le=10, description="抓取页数")

    @field_validator("product_url")
    @classmethod
    def _check_url(cls, v):
        if not v.startswith(("http://", "https://")):
            raise ValueError("product_url 必须是完整链接")
        return v


@registry.register(
    name="scrape_reviews",
    description="抓取指定京东商品的评论并入库。会发起外部网络请求，需要人工确认后才执行。",
    args_model=ScrapeArgs,
    requires_confirmation=True,
    tags=["side-effect", "network"],
)
def scrape_reviews(product_url: str, pages: int = 1) -> Dict[str, Any]:
    from scraper_api import extract_product_id

    pid = extract_product_id(product_url)
    if not pid:
        raise ToolError(f"无法从链接提取商品 ID: {product_url}", retryable=False)
    raise ToolError(
        "爬虫需要本机 Chrome（DrissionPage），当前环境不可用。"
        "可先用 search_reviews / aggregate_reviews 在已有评论库上工作。",
        retryable=False,
    )


# --------------------------------------------------------------------------
# train_model / list_experiments —— 复用毕设已有能力
# --------------------------------------------------------------------------
class TrainArgs(BaseModel):
    model: Literal["cnn", "lstm", "bilstm", "lstm_attention", "bilstm_attention"]
    epochs: int = Field(6, ge=1, le=100)
    batch_size: int = Field(128, ge=8, le=512)
    learning_rate: float = Field(1e-3, gt=0, le=1.0)


@registry.register(
    name="train_model",
    description="按给定超参训练一个本地模型。耗时长且占用算力，需要人工确认后才执行。",
    args_model=TrainArgs,
    requires_confirmation=True,
    tags=["side-effect", "compute"],
)
def train_model(model: str, epochs: int = 6, batch_size: int = 128,
                learning_rate: float = 1e-3) -> Dict[str, Any]:
    cmd = [sys.executable, os.path.join(Config.base_dir, "main.py"),
           "--model", model, "--epochs", str(epochs),
           "--batch-size", str(batch_size), "--learning-rate", str(learning_rate)]
    proc = subprocess.Popen(cmd, cwd=Config.base_dir,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return {"started": True, "pid": proc.pid, "model": model, "epochs": epochs,
             "note": "训练已在后台启动，用 list_experiments 查看进度"}


class ExperimentsArgs(BaseModel):
    model: Optional[Literal["cnn", "lstm", "bilstm", "lstm_attention", "bilstm_attention"]] = None


@registry.register(
    name="list_experiments",
    description="列出各模型在测试集上的真实评测指标（accuracy / F1 / recall / 单条延迟）。",
    args_model=ExperimentsArgs,
    tags=["analysis"],
)
def list_experiments(model: Optional[str] = None) -> Dict[str, Any]:
    from metrics_store import load_all_metrics

    data = load_all_metrics()
    if model:
        data = {k: v for k, v in data.items() if k == model}
    if not data:
        raise ToolError("还没有任何评测记录，请先运行 evaluate.py", retryable=False)
    return {"metrics": data}


def bootstrap_store() -> int:
    """首次使用时把数据集里的真实评论灌进检索库。"""
    return store.seed_from_dataset()
