"""数据约束；字符位置采用从零开始、包含起点但不包含终点的区间。"""

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class Paper:
    paper_id: str
    title: str
    abstract: str
    full_text: str
    abstract_start: int
    source: str = ""
    abstract_is_excerpt: bool = False

    def __post_init__(self):
        for name in ("paper_id", "title", "abstract", "full_text"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"论文字段 {name} 必须是非空文字")
        if type(self.abstract_start) is not int or self.abstract_start < 0:
            raise ValueError("摘要起点必须是非负整数")
        stop = self.abstract_start + len(self.abstract)
        if self.full_text[self.abstract_start:stop] != self.abstract:
            raise ValueError("摘要必须与全文对应位置完全一致")
        if not isinstance(self.source, str) or type(self.abstract_is_excerpt) is not bool:
            raise ValueError("来源必须是文字，节选标记必须是布尔值")


@dataclass(frozen=True)
class Passage:
    start: int
    text: str


@dataclass(frozen=True)
class Evidence:
    paper_id: str
    title: str
    question: str
    answer: str
    quote: str
    start: int
    end: int
    score: float
    phase: str
    source: str


@dataclass
class Assessment:
    paper: Paper
    question: str
    phase: str
    score: float
    evidence: list[Evidence] = field(default_factory=list)


def parse_object(raw: str) -> dict:
    """允许完整 JSON 或单个完整代码围栏，拒绝夹杂解释的响应。"""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) < 3 or lines[0] not in ("```", "```json") or lines[-1] != "```":
            raise ValueError("模型响应的代码围栏格式不合法")
        text = "\n".join(lines[1:-1])
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("模型必须返回 JSON 对象")
    return value


def numeric_score(value) -> float:
    if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 100:
        raise ValueError("评分必须是 0 到 100 的有限数值")
    return float(value)


def load_corpus(path: str | Path) -> list[Paper]:
    papers = []
    seen = set()
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                paper = Paper(**json.loads(line))
                if paper.paper_id in seen:
                    raise ValueError("论文 ID 重复")
            except (TypeError, ValueError) as exc:
                raise ValueError(f"语料第 {number} 行不合法：{exc}") from exc
            seen.add(paper.paper_id)
            papers.append(paper)
    if not papers:
        raise ValueError("语料中没有论文")
    return papers


def as_record(value) -> dict:
    return asdict(value)
