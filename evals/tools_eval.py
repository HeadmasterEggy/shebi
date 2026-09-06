# -*- coding: utf-8 -*-
"""工具层三维对比：本地自训练模型 vs LLM zero-shot。

这是「为什么保留毕设那个模型」的技术论证。没有这张表，简历上写"把自训练
模型封装为 agent 工具"就只是一句自我感动——凭什么不直接让大模型判？

三个维度，缺一不可：

    准确率   同一批 test.txt 样本，同一套标签
    成本     LLM 按实际消耗的 token 折算；本地模型只有电费，记 0
    延迟     p50 / p95。均值会被少数慢请求带偏，agent 里真正卡人的是 p95

两个容易被追问的口径问题，先写在这里：

1. **本地模型的延迟是逐条测的，LLM 是批量测的。** 逐条调 LLM 判 6000 条既
   不现实也不是任何人的真实用法，所以 LLM 走批量、延迟按批摊到每条。这对
   LLM 是有利的口径，但它就是真实用法，报告里注明即可。
2. **本地模型在自己的数据集上评测。** 它是在这批数据的训练集上训的，LLM
   没见过——这个不对等对 LLM 不利，也要写明。结论只能说"在这个领域的数据上
   本地模型不输"，不能推广成"小模型优于大模型"。

没有 API key 时只跑本地半边，LLM 那几格留空，不编数。
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agent.llm import LLMClient, Usage, estimate_cost
from config import Config

SEED = 20260907

SYSTEM = ("你是中文电商评论情感分类器。对每条评论判断情感极性。"
          "只输出一行结果，格式为逗号分隔的数字：1 表示积极，0 表示消极，"
          "顺序与输入编号一致，不要输出任何其他内容。")


@dataclass
class Metrics:
    name: str
    n: int = 0
    correct: int = 0
    latencies_ms: List[float] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    cost_usd: float = 0.0
    unparsed: int = 0

    @property
    def accuracy(self) -> Optional[float]:
        return None if not self.n else round(self.correct / self.n * 100, 2)

    def pct(self, q: float) -> Optional[float]:
        if not self.latencies_ms:
            return None
        s = sorted(self.latencies_ms)
        i = min(len(s) - 1, int(round(q * (len(s) - 1))))
        return round(s[i], 2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "n": self.n,
            "accuracy": self.accuracy,
            "p50_ms": self.pct(0.50),
            "p95_ms": self.pct(0.95),
            "total_tokens": self.usage.total or None,
            "cost_usd": round(self.cost_usd, 6) if self.cost_usd else 0.0,
            # 按每千条折算，比"单条 0.000021 美元"这种数好读
            "cost_per_1k": (round(self.cost_usd / self.n * 1000, 4)
                            if self.n and self.cost_usd else 0.0),
            "unparsed": self.unparsed,
        }


def load_samples(n: int = 500, path: Optional[str] = None) -> List[Dict[str, Any]]:
    """从 test.txt 抽样。文件是「标签 空格分词文本」格式，这里还原成可读文本。"""
    rows = []
    with open(path or Config.test_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            rows.append({"label": int(parts[0]), "text": "".join(parts[1:])})
    rng = random.Random(SEED)
    rng.shuffle(rows)
    return rows[:n]


# ---------------------------------------------------------------------------
def eval_local(samples: List[Dict[str, Any]], model: Optional[str] = None,
               latency_probe: int = 200) -> Metrics:
    """本地模型：批量测准确率，逐条测延迟。

    准确率和延迟分开跑，是因为两者要的口径不同——准确率要全量、延迟要单条。
    用一次批量前向去算"平均单条延迟"会把结果说得比真实单请求好得多。
    """
    from agent.tools import classify_sentiment

    m = Metrics(name=f"本地 {(model or Config.default_model).upper()}")
    out = classify_sentiment([s["text"] for s in samples], model=model)
    for s, r in zip(samples, out["results"]):
        m.n += 1
        if (1 if r["sentiment"] == "积极" else 0) == s["label"]:
            m.correct += 1

    for s in samples[:latency_probe]:
        t0 = time.perf_counter()
        classify_sentiment([s["text"]], model=model)
        m.latencies_ms.append((time.perf_counter() - t0) * 1000)
    return m


def eval_llm(samples: List[Dict[str, Any]], client: LLMClient,
             batch_size: int = 20) -> Metrics:
    """LLM zero-shot：批量提问，按批测延迟再摊到每条。"""
    m = Metrics(name=f"LLM zero-shot（{client.model}）")

    for start in range(0, len(samples), batch_size):
        batch = samples[start:start + batch_size]
        prompt = "\n".join(f"{i + 1}. {s['text']}" for i, s in enumerate(batch))
        t0 = time.perf_counter()
        resp = client.chat([{"role": "system", "content": SYSTEM},
                            {"role": "user", "content": prompt}])
        elapsed = (time.perf_counter() - t0) * 1000

        m.usage = m.usage + resp.usage
        m.cost_usd += estimate_cost(client.model, resp.usage)
        m.latencies_ms.extend([elapsed / len(batch)] * len(batch))

        preds = _parse_labels(resp.content, len(batch))
        for s, p in zip(batch, preds):
            m.n += 1
            if p is None:
                m.unparsed += 1
            elif p == s["label"]:
                m.correct += 1
    return m


def _parse_labels(content: Optional[str], expected: int) -> List[Optional[int]]:
    """把模型回的一行数字解析成标签。

    解析不出来的位置返回 None 并单独计数，不当成答错——「模型不听格式」和
    「模型判错」是两个不同的问题，混在一个准确率里就看不出是哪个。
    """
    import re
    nums = [int(x) for x in re.findall(r"[01]", content or "")]
    out: List[Optional[int]] = list(nums[:expected])
    out += [None] * (expected - len(out))
    return out


# ---------------------------------------------------------------------------
def run(n: int = 500, model: Optional[str] = None, with_llm: bool = False,
        batch_size: int = 20) -> Dict[str, Any]:
    samples = load_samples(n)
    rows = [eval_local(samples, model=model).to_dict()]
    note = None

    if with_llm:
        try:
            from agent.llm import build_client
            rows.append(eval_llm(samples, build_client(), batch_size).to_dict())
        except RuntimeError as e:
            note = f"LLM 半边未运行：{e}"
    else:
        note = "LLM 半边未运行：需要 --llm 且配置 AGENT_API_KEY"

    return {"samples": len(samples), "batch_size": batch_size,
            "rows": rows, "note": note}
