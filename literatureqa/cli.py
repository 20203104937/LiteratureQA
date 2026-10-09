"""命令行入口；真实接口与离线演示由显式参数区分。"""

import argparse
import asyncio
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

from .demo import demo_question
from .ingest import ingest_markdown, ingest_pdf
from .metrics import recall_at_k
from .service import answer_question, build_retriever
from .settings import load_settings


def _write(value: dict, output: str | None):
    text = json.dumps(value, ensure_ascii=False, indent=2)
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def parser():
    root = argparse.ArgumentParser(description="LiteratureQA 文献问答")
    commands = root.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("ingest", help="把 Markdown 或 PDF 导入论文语料")
    ingest.add_argument("input")
    ingest.add_argument("--output", required=True)
    ingest.add_argument("--pdf", action="store_true", help="调用独立安装的 PDF 解析器")
    ingest.add_argument("--parser", default="mineru")
    ingest.add_argument("--work-dir", default="data/parsed")
    ask = commands.add_parser("ask", help="回答一个问题")
    ask.add_argument("--corpus", required=True)
    ask.add_argument("--question", required=True)
    ask.add_argument("--config")
    ask.add_argument("--mode", choices=("bm25", "semantic", "hybrid"))
    ask.add_argument("--output")
    ask.add_argument("--demo", action="store_true", help="离线固定规则演示；不会调用模型")
    evaluate = commands.add_parser("evaluate-retrieval", help="用标准相关论文 ID 评测检索")
    evaluate.add_argument("--corpus", required=True)
    evaluate.add_argument("--labels", required=True)
    evaluate.add_argument("--config")
    evaluate.add_argument("--mode", choices=("bm25", "semantic", "hybrid"))
    evaluate.add_argument("--top-k", type=int, default=10)
    evaluate.add_argument("--output")
    return root


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "ingest":
            count = ingest_pdf(args.input, args.output, args.work_dir, args.parser) if args.pdf else ingest_markdown(args.input, args.output)
            print(f"已导入 {count} 篇文献。")
            return 0
        settings = load_settings(args.config)
        if args.mode:
            settings = replace(settings, retrieval=replace(settings.retrieval, mode=args.mode))
        retriever = build_retriever(args.corpus, settings)
        if args.command == "ask":
            call = demo_question if args.demo else answer_question
            result = asyncio.run(call(args.question, retriever, settings))
            _write(result, args.output)
            return 2 if result["status"] == "failed" else 0
        if args.top_k < 1:
            raise ValueError("top-k 必须是正整数")
        corpus_ids = {paper.paper_id for paper in retriever.papers}
        results = []
        with Path(args.labels).open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                sample = json.loads(line)
                question, gold = sample["question"], sample["relevant_ids"]
                if not isinstance(question, str) or not question.strip() or not isinstance(gold, list) or any(not isinstance(item, str) for item in gold):
                    raise ValueError("评测问题或标准相关论文 ID 格式不合法")
                if not set(gold) <= corpus_ids:
                    raise ValueError("标准相关论文必须存在于当前语料")
                started = time.perf_counter()
                retrieved = [paper.paper_id for paper in retriever.search(question, args.top_k)]
                results.append({"question": question, "retrieved_ids": retrieved, "relevant_ids": gold, "recall_at_k": recall_at_k(retrieved, gold), "duration_seconds": time.perf_counter() - started})
        if not results:
            raise ValueError("评测标注为空")
        _write({"mode": settings.retrieval.mode, "top_k": args.top_k, "sample_count": len(results), "mean_recall_at_k": sum(row["recall_at_k"] for row in results) / len(results), "results": results}, args.output)
        return 0
    except (ValueError, OSError, KeyError, ImportError) as exc:
        print(f"执行失败：{exc}", file=sys.stderr)
        return 2
