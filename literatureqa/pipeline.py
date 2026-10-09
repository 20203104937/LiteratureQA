"""拆分、逐篇取证、评分、正文复查、引用汇总。"""

import asyncio
from dataclasses import asdict
import json
import math
import time
from uuid import uuid4

from . import prompts
from .client import CallTrace, ModelError
from .schema import Assessment, Evidence, Passage, numeric_score, parse_object
from .settings import PipelineSettings


def score_thresholds(scores: list[float], n: float, k: float) -> dict:
    if not scores:
        raise ValueError("没有有效分数，不能计算阈值")
    mean = sum(scores) / len(scores)
    deviation = math.sqrt(sum((score - mean) ** 2 for score in scores) / len(scores))
    return {"mean": mean, "population_std": deviation, "high": mean - n * deviation, "low": mean - (n + k) * deviation}


class QAPipeline:
    def __init__(self, retriever, client, settings: PipelineSettings, trace: CallTrace | None = None):
        self.retriever = retriever
        self.client = client
        self.settings = settings
        self.trace = trace or CallTrace()

    async def _object(self, role: str, payload: dict) -> dict:
        system, user = prompts.request(role, payload, self.settings.max_subquestions)
        return parse_object(await self.client.generate(role, system, user))

    async def _assess(self, paper, question: str, passages: list[Passage], phase: str) -> Assessment:
        documents = [{"paper_id": paper.paper_id, "title": paper.title, "text": passage.text} for passage in passages]
        extracted = await self._object("extract", {"question": question, "documents": documents})
        answer = extracted.get("answer")
        quotes = extracted.get("quotes")
        if not isinstance(answer, str) or not isinstance(quotes, list) or any(not isinstance(quote, str) for quote in quotes):
            raise ValueError("证据提取格式不合法")
        if not answer.strip() or not quotes:
            raise ValueError("当前文献没有可验证证据")
        locations = []
        for quote in dict.fromkeys(quotes):
            if not quote.strip():
                raise ValueError("引用原文不能为空")
            found = next(((passage.start + passage.text.find(quote), quote) for passage in passages if quote in passage.text), None)
            if found is None:
                raise ValueError("引用原文不在提供的文献片段内")
            start, quote = found
            if paper.full_text[start:start + len(quote)] != quote:
                raise ValueError("引用字符位置校验失败")
            locations.append(found)
        judged = await self._object("judge", {"question": question, "documents": documents, "answer": answer, "quotes": quotes})
        score = numeric_score(judged.get("score"))
        evidence = [Evidence(paper.paper_id, paper.title, question, answer, quote, start, start + len(quote), score, phase, paper.source) for start, quote in locations]
        return Assessment(paper, question, phase, score, evidence)

    async def ask(self, question: str) -> dict:
        """每次请求创建独立证据表和任务门限；不共享用户问答状态。"""
        if not isinstance(question, str) or not question.strip() or len(question) > 10000:
            raise ValueError("问题必须非空且不超过 10000 字符")
        started = time.perf_counter()
        errors, decisions, kept = [], [], []
        result = {"request_id": uuid4().hex, "question": question, "status": "insufficient_evidence", "answer": "证据不足，无法给出有依据的回答。", "references": [], "errors": errors, "decisions": decisions}
        gate = asyncio.Semaphore(self.settings.paper_concurrency)

        def failure(stage, query, paper_id, exc):
            errors.append({"stage": stage, "question": query, "paper_id": paper_id, "message": str(exc) if isinstance(exc, (ValueError, ModelError)) else "处理失败，请检查服务或语料"})

        async def assess(paper, query, phase):
            async with gate:
                try:
                    if phase == "abstract":
                        passages = [Passage(paper.abstract_start, paper.abstract[:self.settings.max_abstract_chars])]
                    else:
                        passages = await asyncio.to_thread(self.retriever.passages, paper, query, self.settings.passage_count, self.settings.passage_chars, self.settings.passage_overlap)
                    if not passages:
                        raise ValueError("没有检索到正文片段")
                    return await self._assess(paper, query, passages, phase)
                except Exception as exc:
                    failure(phase, query, paper.paper_id, exc)
                    return None

        async def process(query):
            try:
                papers = await asyncio.to_thread(self.retriever.search, query, self.settings.top_k)
            except Exception as exc:
                failure("retrieval", query, "", exc)
                return
            assessed = await asyncio.gather(*(assess(paper, query, "abstract") for paper in papers))
            valid = [item for item in assessed if item is not None]
            report = {"question": query, "candidate_ids": [paper.paper_id for paper in papers], "abstract_scores": {item.paper.paper_id: item.score for item in valid}, "accepted_ids": [], "reviewed_ids": [], "fulltext_scores": {}}
            decisions.append(report)
            if not valid:
                return
            thresholds = score_thresholds([item.score for item in valid], self.settings.threshold_n, self.settings.threshold_k)
            report["thresholds"] = thresholds
            accepted = [item for item in valid if item.score >= thresholds["high"]]
            review = [item for item in valid if thresholds["low"] <= item.score < thresholds["high"]]
            report["reviewed_ids"] = [item.paper.paper_id for item in review]
            reviewed = await asyncio.gather(*(assess(item.paper, query, "fulltext") for item in review))
            for item in reviewed:
                if item is not None:
                    report["fulltext_scores"][item.paper.paper_id] = item.score
                    if item.score > thresholds["high"]:
                        accepted.append(item)
            report["accepted_ids"] = [item.paper.paper_id for item in accepted]
            kept.extend(evidence for item in accepted for evidence in item.evidence)

        try:
            split = await self._object("decompose", {"question": question})
            questions = split.get("questions")
            if not isinstance(questions, list) or not 1 <= len(questions) <= self.settings.max_subquestions or any(not isinstance(item, str) or not item.strip() or len(item) > 10000 for item in questions):
                raise ValueError("问题拆分格式不合法")
            questions = list(dict.fromkeys(item.strip() for item in questions))
            result["subquestions"] = questions
            await asyncio.gather(*(process(query) for query in questions))
            if kept:
                # 稳定排序使引用号不随并发完成顺序变化。
                unique = {}
                for item in sorted(kept, key=lambda item: (item.paper_id, item.start, item.question)):
                    unique.setdefault((item.paper_id, item.start, item.end, item.question), item)
                selected, used_chars = [], 0
                for item in unique.values():
                    row = {"number": len(selected) + 1, **asdict(item)}
                    chars = len(json.dumps(row, ensure_ascii=False))
                    if used_chars + chars > self.settings.max_final_evidence_chars:
                        continue
                    selected.append(row)
                    used_chars += chars
                result["retained_evidence_count"] = len(unique)
                result["synthesis_evidence_count"] = len(selected)
                if selected:
                    synthesis = await self._object("synthesize", {"question": question, "evidence": selected})
                    claims = synthesis.get("claims")
                    if not isinstance(claims, list):
                        raise ValueError("最终回答格式不合法")
                    paragraphs, cited = [], set()
                    for claim in claims:
                        if not isinstance(claim, dict) or not isinstance(claim.get("text"), str) or not claim["text"].strip():
                            raise ValueError("最终结论必须是非空文字")
                        citations = claim.get("citations")
                        if not isinstance(citations, list) or not citations or any(type(number) is not int or not 1 <= number <= len(selected) for number in citations):
                            raise ValueError("最终回答引用了不存在或缺失的证据编号")
                        citations = sorted(set(citations))
                        cited.update(citations)
                        paragraphs.append(claim["text"].strip() + " " + "".join(f"[{number}]" for number in citations))
                    if paragraphs:
                        result.update(status="ok", answer="\n\n".join(paragraphs), references=[row for row in selected if row["number"] in cited])
        except Exception as exc:
            failure("pipeline", question, "", exc)
            result["status"] = "failed"
            result["answer"] = "本次问答失败；请查看 errors 字段。"
        result["decisions"] = sorted(decisions, key=lambda item: item["question"])
        result["duration_seconds"] = time.perf_counter() - started
        result["calls"] = list(self.trace.records)
        return result
