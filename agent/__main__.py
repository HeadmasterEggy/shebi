# -*- coding: utf-8 -*-
"""命令行入口。

    python -m agent "这批评论里差评主要集中在什么问题上？"
    python -m agent --tools           # 列出工具与 JSON Schema
    python -m agent --trace <run_id>  # 回看一次执行轨迹

provider 由环境变量决定（同一份代码切换 OpenAI / DeepSeek / Qwen）：
    AGENT_API_KEY=...  AGENT_BASE_URL=https://api.deepseek.com/v1  AGENT_MODEL=deepseek-chat
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from agent import tools as _tools  # noqa: F401  —— import 即注册
from agent.budget import Budget
from agent.llm import build_client
from agent.react import ReActAgent
from agent.registry import registry
from agent.trace import list_traces, load_trace


def _confirm_interactive(name: str, args: dict) -> bool:
    print(f"\n⚠️  工具 {name} 有副作用，参数：{json.dumps(args, ensure_ascii=False)}")
    return input("放行？[y/N] ").strip().lower() == "y"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m agent")
    ap.add_argument("task", nargs="?", help="要交给 agent 的任务")
    ap.add_argument("--tools", action="store_true", help="列出所有工具及其 JSON Schema")
    ap.add_argument("--trace", metavar="RUN_ID", help="回看一次执行轨迹")
    ap.add_argument("--traces", action="store_true", help="列出已有的 run_id")
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--max-cost", type=float, default=0.50, help="成本上限（美元）")
    ap.add_argument("--yes", action="store_true", help="自动放行有副作用的工具")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    if args.tools:
        for tool in registry.all():
            flag = "  [需人工确认]" if tool.requires_confirmation else ""
            print(f"\n=== {tool.name}{flag} ===")
            print(tool.description)
            print(json.dumps(tool.json_schema()["function"]["parameters"],
                             ensure_ascii=False, indent=2))
        return 0

    if args.traces:
        for rid in list_traces():
            print(rid)
        return 0

    if args.trace:
        data = load_trace(args.trace)
        print(json.dumps(data["summary"], ensure_ascii=False, indent=2))
        for s in data["steps"]:
            head = f"[{s['step']:>2}] {s['kind']:<5}"
            if s["kind"] == "llm":
                print(f"{head} {(s.get('thought') or '')[:140]}")
            elif s["kind"] == "tool":
                mark = "ok " if s.get("ok") else "ERR"
                print(f"{head} {mark} {s.get('tool')} {json.dumps(s.get('tool_args'), ensure_ascii=False)[:100]}")
                print(f"        → {(s.get('observation') or '')[:200]}")
            else:
                print(f"{head} {(s.get('thought') or s.get('error') or '')[:200]}")
        return 0

    if not args.task:
        ap.print_help()
        return 2

    _tools.bootstrap_store()

    agent = ReActAgent(
        client=build_client(),
        budget=Budget(max_steps=args.max_steps, max_cost_usd=args.max_cost),
        confirm=(lambda n, a: True) if args.yes else _confirm_interactive,
    )
    result = agent.run(args.task)
    path = result.trace.save()

    print("\n" + "=" * 60)
    print(result.answer or "(无答案)")
    print("=" * 60)
    print(json.dumps(result.trace.summary(), ensure_ascii=False, indent=2))
    print(f"轨迹已保存: {path}")
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
