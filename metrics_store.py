# -*- coding: utf-8 -*-
"""按模型分开保存的真实评测指标。

数据来源只有一个：evaluate.py 在 data/test.txt 上跑出来的结果。
读不到就返回 None —— 界面上宁可显示「未评测」，也不展示编造的数字。
"""

import json
import logging
import os
from datetime import datetime, timezone

from config import Config

logger = logging.getLogger(__name__)

METRICS_PATH = os.path.join(Config.runtime_dir, "model_metrics.json")


def _read_all():
    if not os.path.exists(METRICS_PATH):
        return {}
    try:
        with open(METRICS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.error("读取评测指标失败 %s: %s", METRICS_PATH, e)
        return {}


def load_model_metrics(model_type=None):
    """返回该模型的评测指标，没有记录时返回 None。"""
    model_type = (model_type or Config.default_model).lower()
    entry = _read_all().get(model_type)
    if not entry:
        return None
    return {
        "accuracy": entry["accuracy"],
        "f1_score": entry["f1_score"],
        "recall": entry["recall"],
        "evaluatedAt": entry.get("evaluated_at"),
        "sampleCount": entry.get("sample_count"),
        "dataset": entry.get("dataset", "test"),
    }


def save_model_metrics(model_type, accuracy, f1_score, recall,
                       sample_count=None, dataset="test", extra=None):
    """写入一个模型的评测结果（比例值，0–1）。"""
    model_type = model_type.lower()
    data = _read_all()
    entry = {
        "accuracy": round(float(accuracy), 6),
        "f1_score": round(float(f1_score), 6),
        "recall": round(float(recall), 6),
        "sample_count": sample_count,
        "dataset": dataset,
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if extra:
        entry.update(extra)
    data[model_type] = entry

    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
    logger.info("已写入 %s 的评测指标 -> %s", model_type, METRICS_PATH)
    return entry


def load_all_metrics():
    return _read_all()
