# -*- coding: utf-8 -*-
"""在测试集上评测模型，产出真实指标。

用法:
    python evaluate.py                    # 评测所有有权重的模型
    python evaluate.py --model bilstm     # 只评测一个
    python evaluate.py --table            # 只打印已有结果表

结果写入 runtime/model_metrics.json，由 Web 端读取展示。
这是界面上 accuracy / F1 / recall 的唯一来源。
"""

import argparse
import logging
import os
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torch.utils.data import DataLoader

from config import Config
from data_Process import Data_set, text_to_array
from inference import engine
from metrics_store import save_model_metrics, load_all_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def evaluate_model(model_type, batch_size=256):
    """在 data/test.txt 上评测，返回指标字典。"""
    word2id = engine.word2id
    arr, labels = text_to_array(word2id, Config.max_sen_len, Config.test_path)
    labels = np.array(labels).reshape(-1, 1)

    model = engine.get_model(model_type)
    model.eval()

    loader = DataLoader(Data_set(arr, labels), batch_size=batch_size, shuffle=False)

    preds, golds, batch_times = [], [], []
    with torch.no_grad():
        for x, y in loader:
            t0 = time.perf_counter()
            out = model(x.long())
            batch_times.append((time.perf_counter() - t0) / x.shape[0])
            preds.extend(out.argmax(1).tolist())
            golds.extend(y.squeeze(1).tolist())

    preds, golds = np.array(preds), np.array(golds)
    acc = accuracy_score(golds, preds)
    f1 = f1_score(golds, preds, average="weighted")
    rec = recall_score(golds, preds, average="weighted")

    # 预测分布是判断模型是否坍缩的最直接信号：
    # 如果全部落在一个类上，说明推理通路有问题，而不是模型"学得不好"。
    dist = np.bincount(preds, minlength=Config.n_class).tolist()
    collapsed = bool(min(dist) == 0)

    per_sample_ms = float(np.mean(batch_times) * 1000)

    save_model_metrics(
        model_type, acc, f1, rec,
        sample_count=int(len(golds)),
        extra={
            "prediction_distribution": dist,
            "collapsed": collapsed,
            "per_sample_ms": round(per_sample_ms, 4),
        },
    )

    return {
        "model": model_type, "accuracy": acc, "f1": f1, "recall": rec,
        "dist": dist, "collapsed": collapsed, "per_sample_ms": per_sample_ms,
        "n": len(golds),
    }


def available_models():
    out = []
    for m in Config.model_choices:
        path = os.path.join(Config.model_dir, f"{m}_model_best.pkl")
        # 小于 1KB 的多半是没拉下来的 Git LFS 指针文件
        if os.path.exists(path) and os.path.getsize(path) > 1024:
            out.append(m)
    return out


def print_table(rows):
    head = f"{'模型':<20}{'Accuracy':>10}{'F1':>10}{'Recall':>10}{'预测分布':>16}{'单条延迟':>12}"
    print("\n" + head)
    print("-" * len(head.encode('utf-8').decode('utf-8')) + "-" * 20)
    for r in rows:
        flag = "  ← 坍缩" if r["collapsed"] else ""
        print(f"{r['model']:<20}{r['accuracy']*100:>9.2f}%{r['f1']*100:>9.2f}%"
              f"{r['recall']*100:>9.2f}%{str(r['dist']):>16}{r['per_sample_ms']:>10.2f}ms{flag}")
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=Config.model_choices, help="只评测指定模型")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--table", action="store_true", help="只打印已保存的结果")
    args = ap.parse_args()

    if args.table:
        saved = load_all_metrics()
        rows = [{"model": k, "accuracy": v["accuracy"], "f1": v["f1_score"],
                 "recall": v["recall"], "dist": v.get("prediction_distribution", []),
                 "collapsed": v.get("collapsed", False),
                 "per_sample_ms": v.get("per_sample_ms", 0), "n": v.get("sample_count")}
                for k, v in sorted(saved.items())]
        print_table(rows)
        raise SystemExit(0)

    targets = [args.model] if args.model else available_models()
    if not targets:
        raise SystemExit("没有可评测的模型权重，请先运行 main.py 训练。")

    logger.info("待评测模型: %s", targets)
    results = []
    for m in targets:
        logger.info("正在评测 %s ...", m)
        results.append(evaluate_model(m, args.batch_size))

    print_table(results)
