# -*- coding: utf-8 -*-
"""评测入口。

    python -m evals                    # 全部能离线跑的（Critic + 工具层本地半边）
    python -m evals critic             # 只跑 Critic 评测
    python -m evals tools --llm        # 工具层三维对比，含 LLM（需要 AGENT_API_KEY）
    python -m evals agent --limit 5    # Agent 任务集（需要 AGENT_API_KEY）
    python -m evals --md report.md     # 把结果写成 Markdown
"""

from __future__ import annotations

import argparse
import json
import logging
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m evals")
    ap.add_argument("suite", nargs="?", default="offline",
                    choices=["offline", "critic", "tools", "agent", "all"],
                    help="跑哪一套；offline（默认）只跑不需要 API key 的部分")
    ap.add_argument("-n", "--samples", type=int, default=500, help="工具层评测的抽样条数")
    ap.add_argument("--per-kind", type=int, default=40, help="Critic 评测每类幻觉的样本数")
    ap.add_argument("--llm", action="store_true", help="工具层评测带上 LLM 半边")
    ap.add_argument("--batch-size", type=int, default=20, help="LLM 批量条数")
    ap.add_argument("--limit", type=int, help="Agent 评测只跑前 N 条任务")
    ap.add_argument("--kind", help="Agent 评测只跑某一类任务")
    ap.add_argument("--md", metavar="PATH", help="把 Markdown 报告写到文件")
    ap.add_argument("--json", metavar="PATH", help="把原始结果写到 JSON")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.ERROR,
                        format="%(levelname)s %(name)s: %(message)s")
    if not args.verbose:
        logging.disable(logging.INFO)

    from evals import agent_eval, critic_eval, report, tools_eval

    critic = tools = agent = None
    want = args.suite

    if want in ("offline", "critic", "all"):
        print("跑 Critic 评测……", file=sys.stderr)
        critic = critic_eval.run(per_kind=args.per_kind)

    if want in ("offline", "tools", "all"):
        print("跑工具层评测……", file=sys.stderr)
        tools = tools_eval.run(n=args.samples,
                               with_llm=args.llm or want == "all",
                               batch_size=args.batch_size)

    if want in ("agent", "all"):
        print("跑 Agent 任务集……", file=sys.stderr)
        try:
            agent = agent_eval.run(limit=args.limit, kind=args.kind)
        except RuntimeError as e:
            print(f"Agent 评测跳过：{e}", file=sys.stderr)

    md = report.render(critic=critic, tools=tools, agent=agent)
    print(md)

    if args.md:
        with open(args.md, "w", encoding="utf-8") as f:
            f.write(md + "\n")
        print(f"\nMarkdown 已写入 {args.md}", file=sys.stderr)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"critic": critic, "tools": tools, "agent": agent},
                      f, ensure_ascii=False, indent=2, default=str)
        print(f"原始结果已写入 {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
