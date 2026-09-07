# -*- coding: utf-8 -*-
"""把评测结果渲染成可以直接贴进 README 的 Markdown。

刻意不做成"跑完自动改写 README"——那样 README 里的数字会在没人看着的时候
悄悄变化。这里只输出文本，贴不贴、贴哪一版，由人决定。

空值一律渲染成 `—` 而不是 0：没测和测出来是 0，是两件事。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from evals.critic_eval import KINDS as CRITIC_KINDS

KIND_LABELS = {
    "grounded": "真有据<br><sub>不该误伤</sub>",
    "fabricated_id": "编造 id",
    "no_citation": "无引用",
    "wrong_polarity": "极性错配",
    "wrong_aspect": "方面错配",
}


def _n(v: Any, suffix: str = "") -> str:
    if v is None:
        return "—"
    if isinstance(v, list):
        return " / ".join(str(x) for x in v)
    return f"{v}{suffix}"


def _table(headers: List[str], rows: List[List[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def critic_md(data: Dict[str, Any]) -> str:
    if not data.get("cases"):
        return f"_Critic 评测未运行：{data.get('error', '未知原因')}_"

    counts = data["per_kind_counts"]
    headers = ["配置"] + [f"{KIND_LABELS[k]}<br><sub>n={counts.get(k, 0)}</sub>"
                          for k in CRITIC_KINDS]
    rows = [[r["name"]] + [_n(r[k], "%") for k in CRITIC_KINDS] for r in data["results"]]

    return "\n".join([
        f"对照集 {data['cases']} 条，四类幻觉各自对应 Critic 的一道关卡。",
        "",
        _table(headers, rows),
        "",
        "**哪几个数字算成绩，哪几个不算：**",
        "",
        "- `编造 id` / `无引用` 是结构性的，查库和正则必然全中——它们在表里是为了"
        "证明关卡接线正确，不是能力证明。",
        "- `方面错配` 的负例是用方面关卡自己那份词典构造的，**属于自证**，"
        "只能说明关卡按设计工作，不能当泛化能力。",
        "- **只有 `极性错配` 是真测量**：负例用数据集的 gold_label 构造，"
        "关卡用的是本地模型的预测，两边独立。",
        "- `真有据` 一列是误伤率的反面。它低于 100% 的部分，就是被错误打回、"
        "要多改一轮的代价。",
    ])


def tools_md(data: Dict[str, Any]) -> str:
    headers = ["", "准确率", "p50 延迟", "p95 延迟", "成本 / 千条", "格式失败"]
    rows = []
    for r in data["rows"]:
        rows.append([
            r["name"], _n(r["accuracy"], "%"), _n(r["p50_ms"], " ms"),
            _n(r["p95_ms"], " ms"),
            "≈ $0" if not r["cost_per_1k"] else f"${r['cost_per_1k']}",
            _n(r["unparsed"]) if r["unparsed"] else "—",
        ])
    lines = [f"同一批 {data['samples']} 条 test.txt 样本，同一套标签。", "",
             _table(headers, rows), "",
             "**口径说明（会被追问，先写清楚）：**", "",
             f"- 本地模型的延迟是**逐条**测的；LLM 走批量（每批 {data['batch_size']} 条）"
             "再摊到每条。逐条调 LLM 判几千条不是任何人的真实用法，"
             "但这个口径对 LLM 有利，得注明。",
             "- 本地模型是在这批数据的训练集上训的，LLM 没见过。这个不对等对 LLM 不利。"
             "所以结论只能说「在这个领域的数据上本地模型不输」，"
             "**不能**推广成「小模型优于大模型」。"]
    if data.get("note"):
        lines += ["", f"> {data['note']}"]
    return "\n".join(lines)


def agent_md(data: Optional[Dict[str, Any]]) -> str:
    if not data:
        return ("_Agent 层评测未运行：需要 `AGENT_API_KEY`。_\n\n"
                "任务集与评测口径已就绪（`evals/tasks.py`，40 条，其中 12 条拒答题），"
                "配好 key 后 `python -m evals agent` 直接出表。")

    s = data["summary"]
    rows = []
    if s.get("执行失败"):
        rows.append(["**执行失败**", f"{s['执行失败']} / {s['tasks']}",
                     f"基础设施问题（网络 / 余额），不计入下面的质量指标："
                     f"{', '.join(s.get('失败任务', [])[:8])}"
                     + ("…" if len(s.get("失败任务", [])) > 8 else "")])
    rows += [
        ["任务完成率", _n(s["任务完成率"], "%"), "非拒答题里产出可用答案的比例"],
        ["拒答正确率", _n(s["拒答正确率"], "%"), "库里没有的信息，确实拒答的比例"],
        ["工具调用正确率", _n(s["工具调用正确率"], "%"), "该用的能力用上了没有"],
        ["引用准确率", _n(s["引用准确率"], "%"), f"结论 {_n(s['结论数'])}，由 Critic 确定性判定"],
        ["平均步数", _n(s["平均步数"]), ""],
        ["平均 token", _n(s["平均 token"]), ""],
        ["单任务成本", f"${_n(s['单任务成本 USD'])}", f"全集共 ${_n(s['总成本 USD'])}"],
        ["p50 / p95 延迟", f"{_n(s['p50 延迟 ms'])} / {_n(s['p95 延迟 ms'])} ms", ""],
    ]
    head = (f"{s['tasks']} 条任务，模型 `{data['model']}`，"
            f"其中 {s.get('执行成功', s['tasks'])} 条执行成功。")
    return "\n".join([head, "", _table(["指标", "结果", "说明"], rows)])


def render(critic: Optional[Dict[str, Any]] = None,
           tools: Optional[Dict[str, Any]] = None,
           agent: Optional[Dict[str, Any]] = None) -> str:
    parts = ["# 评测报告", ""]
    if tools is not None:
        parts += ["## 工具层：本地模型 vs LLM zero-shot", "", tools_md(tools), ""]
    if critic is not None:
        parts += ["## Critic：抓幻觉的能力与代价", "", critic_md(critic), ""]
    parts += ["## Agent 层：40 条任务集", "", agent_md(agent), ""]
    return "\n".join(parts)
