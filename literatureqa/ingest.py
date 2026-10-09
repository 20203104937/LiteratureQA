"""导入 Markdown；PDF 交给用户独立安装的解析工具。"""

from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from uuid import uuid4

from .schema import Paper


def paper_from_markdown(text: str, paper_id: str, source: str) -> Paper:
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        raise ValueError("Markdown 内容为空")
    title_match = re.search(r"^#\s+(.+)$", text, re.M)
    title = title_match.group(1).strip() if title_match else Path(source).stem
    heading = re.search(r"^#{1,6}\s+(?:abstract|摘要)\s*\n", text, re.I | re.M)
    excerpt = heading is None
    start = heading.end() if heading else (title_match.end() if title_match else 0)
    next_heading = re.search(r"^#{1,6}\s+", text[start:], re.M)
    stop = start + next_heading.start() if next_heading else len(text)
    if excerpt:
        stop = min(stop, start + 3000)
    while start < stop and text[start].isspace():
        start += 1
    while stop > start and text[stop - 1].isspace():
        stop -= 1
    if stop <= start:
        raise ValueError("摘要或开头节选为空，请补充内容")
    return Paper(paper_id, title, text[start:stop], text, start, source, excerpt)


def write_corpus(papers: list[Paper], output: str | Path):
    """全部成功后才替换输出，失败时保留已有语料。"""
    if not papers or len({paper.paper_id for paper in papers}) != len(papers):
        raise ValueError("语料必须非空且论文 ID 唯一")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output.parent, delete=False) as stream:
            temporary = Path(stream.name)
            for paper in papers:
                stream.write(json.dumps(asdict(paper), ensure_ascii=False) + "\n")
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def ingest_markdown(input_path: str | Path, output: str | Path) -> int:
    root = Path(input_path)
    files = sorted(root.rglob("*.md")) if root.is_dir() else [root]
    papers = []
    for path in files:
        if path.suffix.lower() != ".md":
            raise ValueError("请输入 Markdown 文件或目录")
        source = path.relative_to(root).as_posix() if root.is_dir() else path.name
        papers.append(paper_from_markdown(path.read_text(encoding="utf-8"), f"paper:{source}", source))
    write_corpus(papers, output)
    return len(papers)


def ingest_pdf(input_path: str | Path, output: str | Path, work_dir: str | Path, executable: str = "mineru", timeout: float = 1800) -> int:
    root = Path(input_path)
    files = sorted(root.rglob("*.pdf")) if root.is_dir() else [root]
    if not files:
        raise ValueError("没有找到 PDF")
    work = Path(work_dir) / uuid4().hex
    work.mkdir(parents=True)
    papers = []
    for number, path in enumerate(files, 1):
        if path.suffix.lower() != ".pdf":
            raise ValueError("请输入 PDF 文件或目录")
        destination = work / str(number)
        destination.mkdir()
        with (destination / "parser.log").open("w", encoding="utf-8") as log:
            subprocess.run([executable, "-p", str(path.resolve()), "-o", str(destination.resolve())], stdout=log, stderr=subprocess.STDOUT, check=True, timeout=timeout, shell=False)
        markdown = sorted(destination.rglob("*.md"))
        if len(markdown) != 1:
            raise ValueError("解析结果必须恰好包含一个 Markdown 文件，请检查解析器输出")
        source = path.relative_to(root).as_posix() if root.is_dir() else path.name
        papers.append(paper_from_markdown(markdown[0].read_text(encoding="utf-8"), f"paper:{source}", source))
    write_corpus(papers, output)
    return len(papers)
