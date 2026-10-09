import asyncio
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from literatureqa.client import CallTrace, ChatClient, ModelError
from literatureqa.ingest import ingest_markdown, ingest_pdf, paper_from_markdown, write_corpus
from literatureqa.metrics import candidate_retention, recall_at_k, speedup
from literatureqa.pipeline import QAPipeline, score_thresholds
from literatureqa.retrieval import CorpusRetriever, KeywordIndex, fuse_rankings, make_passages
from literatureqa.schema import Paper, load_corpus, numeric_score, parse_object
from literatureqa.settings import ModelSettings, PipelineSettings, RetrievalSettings, load_settings


def paper(number):
    abstract = f"关键词 paper{number} 的摘要原文。"
    full = f"# 文献 {number}\n" + abstract + f"\n正文说明 paper{number} 的关键词证据。"
    return Paper(f"p{number}", f"文献{number}", abstract, full, full.index(abstract), f"p{number}.md")


class ScriptedClient:
    def __init__(self, scores=None, reviewed=61, invalid_quote=False, invalid_citation=False, split=None):
        self.scores = scores or {"p1": 80, "p2": 60, "p3": 40}
        self.reviewed = reviewed
        self.invalid_quote = invalid_quote
        self.invalid_citation = invalid_citation
        self.split = split
        self.events = []
        self.active = 0
        self.maximum = 0

    async def generate(self, role, system, user):
        data = json.loads(user)
        self.events.append((role, data))
        self.active += 1
        self.maximum = max(self.maximum, self.active)
        try:
            await asyncio.sleep(0)
            if role == "decompose":
                result = {"questions": self.split if self.split is not None else [data["question"]]}
            elif role == "extract":
                text = data["documents"][0]["text"]
                result = {"answer": "基于当前文献的回答", "quotes": ["不存在的原文" if self.invalid_quote else text]}
            elif role == "judge":
                document = data["documents"][0]
                score = self.reviewed if document["text"].startswith("# ") else self.scores[document["paper_id"]]
                result = {"score": score}
            else:
                result = {"claims": [{"text": "文献提供了关键词证据。", "citations": [999] if self.invalid_citation else [row["number"] for row in data["evidence"]]}]}
            return json.dumps(result, ensure_ascii=False)
        finally:
            self.active -= 1


class DataTests(unittest.TestCase):
    def test_markdown_abstract_and_offsets(self):
        result = paper_from_markdown("# 标题\r\n\r\n## 摘要\r\n原文。\r\n\r\n## 方法\r\n正文。", "p", "a.md")
        self.assertEqual(result.title, "标题")
        self.assertEqual(result.abstract, "原文。")
        self.assertFalse(result.abstract_is_excerpt)
        self.assertEqual(result.full_text[result.abstract_start:result.abstract_start + len(result.abstract)], result.abstract)

    def test_excerpt_is_marked(self):
        result = paper_from_markdown("# 标题\n\n开头内容。", "p", "a.md")
        self.assertTrue(result.abstract_is_excerpt)

    def test_invalid_offset_rejected(self):
        with self.assertRaises(ValueError):
            Paper("p", "标题", "摘要", "全文", 0)

    def test_corpus_roundtrip_and_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "papers.jsonl"
            write_corpus([paper(1)], output)
            self.assertEqual(load_corpus(output), [paper(1)])
            output.write_text(output.read_text(encoding="utf-8") * 2, encoding="utf-8")
            with self.assertRaises(ValueError):
                load_corpus(output)

    def test_ingest_failure_preserves_previous_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "papers"
            source.mkdir()
            (source / "good.md").write_text("# 标题\n\n摘要文字", encoding="utf-8")
            output = root / "papers.jsonl"
            ingest_markdown(source, output)
            previous = output.read_bytes()
            (source / "empty.md").write_text("", encoding="utf-8")
            with self.assertRaises(ValueError):
                ingest_markdown(source, output)
            self.assertEqual(output.read_bytes(), previous)

    def test_pdf_adapter_with_fake_parser(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.pdf"
            source.write_bytes(b"fake")
            def fake_run(command, **kwargs):
                self.assertFalse(kwargs["shell"])
                destination = Path(command[command.index("-o") + 1])
                (destination / "result.md").write_text("# 解析示例\n\n## 摘要\n虚构摘要", encoding="utf-8")
            with patch("literatureqa.ingest.subprocess.run", side_effect=fake_run):
                self.assertEqual(ingest_pdf(source, root / "papers.jsonl", root / "work"), 1)
            self.assertEqual(load_corpus(root / "papers.jsonl")[0].source, "input.pdf")

    def test_score_validation(self):
        for value in (-1, 101, True, "90", None, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                numeric_score(value)
        self.assertEqual(numeric_score(0), 0)
        self.assertEqual(numeric_score(100), 100)

    def test_json_prose_is_not_accepted(self):
        self.assertEqual(parse_object('```json\n{"score":80}\n```'), {"score": 80})
        with self.assertRaises(ValueError):
            parse_object('分数如下 {"score":80}')


class RetrievalTests(unittest.TestCase):
    def test_keyword_and_no_matches(self):
        index = KeywordIndex(["苹果水果", "数据库 indexing", "苹果苹果"])
        self.assertEqual(index.rank("indexing")[0][0], 1)
        self.assertTrue(index.rank("水果"))
        self.assertEqual(index.rank("unknown"), [])

    def test_rrf_uses_ranks(self):
        result = dict(fuse_rankings([[(0, 1000), (1, 1)], [(1, 9999), (0, 0)]], 60))
        self.assertAlmostEqual(result[0], 1 / 61 + 1 / 62)
        self.assertAlmostEqual(result[0], result[1])

    def test_rrf_duplicate_rejected(self):
        with self.assertRaises(ValueError):
            fuse_rankings([[(0, 1), (0, 2)]], 60)

    def test_chunks_preserve_offsets(self):
        text = "abcdefghijklmnopqrstuvwxyz"
        chunks = make_passages(text, 10, 3)
        self.assertEqual(chunks[1].start, 7)
        for chunk in chunks:
            self.assertEqual(text[chunk.start:chunk.start + len(chunk.text)], chunk.text)
        self.assertEqual(chunks[-1].start + len(chunks[-1].text), len(text))

    def test_invalid_overlap_rejected(self):
        with self.assertRaises(ValueError):
            make_passages("abc", 10, 10)

    def test_hybrid_with_fake_encoder(self):
        class Scores:
            def tolist(self):
                return [1.0, 0.0]
        class Vectors:
            def __matmul__(self, query_vector):
                return Scores()
        class Encoder:
            def encode(self, texts, query=False):
                return [[1.0, 0.0]] if query else Vectors()
        retriever = CorpusRetriever([paper(1), paper(2)], RetrievalSettings(mode="hybrid"), Encoder())
        self.assertEqual(retriever.search("paper1", 1)[0].paper_id, "p1")


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    def pipeline(self, client=None, **overrides):
        settings = replace(PipelineSettings(threshold_n=0, threshold_k=2), **overrides)
        retriever = CorpusRetriever([paper(1), paper(2), paper(3)], RetrievalSettings())
        return QAPipeline(retriever, client or ScriptedClient(), settings)

    async def test_full_process_and_grounded_offsets(self):
        result = await self.pipeline().ask("关键词")
        self.assertEqual(result["status"], "ok")
        report = result["decisions"][0]
        self.assertEqual(report["thresholds"]["high"], 60)
        self.assertEqual(report["reviewed_ids"], ["p3"])
        self.assertEqual(set(report["accepted_ids"]), {"p1", "p2", "p3"})
        by_id = {paper(i).paper_id: paper(i) for i in (1, 2, 3)}
        for reference in result["references"]:
            source = by_id[reference["paper_id"]].full_text
            self.assertEqual(source[reference["start"]:reference["end"]], reference["quote"])

    async def test_review_equality_is_rejected(self):
        result = await self.pipeline(ScriptedClient(reviewed=60)).ask("关键词")
        self.assertEqual(set(result["decisions"][0]["accepted_ids"]), {"p1", "p2"})

    async def test_quote_not_in_source_rejected(self):
        client = ScriptedClient(invalid_quote=True)
        result = await self.pipeline(client).ask("关键词")
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertFalse(result["references"])
        self.assertEqual(len(result["errors"]), 3)
        self.assertFalse(any(role == "judge" for role, _ in client.events))

    async def test_unknown_citation_rejected(self):
        result = await self.pipeline(ScriptedClient(invalid_citation=True)).ask("关键词")
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["references"])

    async def test_invalid_judge_not_replaced(self):
        result = await self.pipeline(ScriptedClient(scores={"p1": "80", "p2": True, "p3": 101})).ask("关键词")
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertEqual(result["decisions"][0]["abstract_scores"], {})

    async def test_no_candidate_does_not_call_extraction(self):
        client = ScriptedClient()
        result = await self.pipeline(client).ask("nonexistentword")
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertEqual([role for role, _ in client.events], ["decompose"])

    async def test_invalid_decomposition_fails(self):
        result = await self.pipeline(ScriptedClient(split=[])).ask("关键词")
        self.assertEqual(result["status"], "failed")

    async def test_concurrency_one_does_not_deadlock(self):
        client = ScriptedClient(split=["关键词", "paper1"])
        result = await asyncio.wait_for(self.pipeline(client, paper_concurrency=1).ask("关键词问题"), 3)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(result["subquestions"]), 2)

    async def test_repeated_questions_do_not_mix_evidence(self):
        first = await self.pipeline().ask("paper1")
        second = await self.pipeline().ask("paper2")
        self.assertEqual({row["paper_id"] for row in first["references"]}, {"p1"})
        self.assertEqual({row["paper_id"] for row in second["references"]}, {"p2"})
        self.assertNotEqual(first["request_id"], second["request_id"])

    async def test_final_budget_does_not_truncate_quote(self):
        result = await self.pipeline(max_final_evidence_chars=1).ask("关键词")
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertEqual(result["synthesis_evidence_count"], 0)


class SettingsAndMetricsTests(unittest.TestCase):
    def test_threshold_population_std(self):
        result = score_thresholds([80, 60, 40], 0.5, 0.5)
        self.assertEqual(result["mean"], 60)
        self.assertAlmostEqual(result["population_std"], math.sqrt(800 / 3))
        self.assertAlmostEqual(result["low"], 60 - math.sqrt(800 / 3))

    def test_role_overrides_and_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            path.write_text('[model]\nmodel="公共模型"\n[roles.judge]\nmodel="评分模型"\n', encoding="utf-8")
            with patch.dict(os.environ, {"LITERATUREQA_BASE_URL": "https://example.invalid/v1"}, clear=True):
                settings = load_settings(path)
            self.assertEqual(settings.models["judge"].model, "评分模型")
            self.assertEqual(settings.models["extract"].model, "公共模型")
            self.assertEqual(settings.models["judge"].base_url, "https://example.invalid/v1")

    def test_unknown_or_invalid_configuration_rejected(self):
        with self.assertRaises(ValueError):
            PipelineSettings(paper_concurrency=0)
        with self.assertRaises(ValueError):
            RetrievalSettings(rrf_constant=0)
        with self.assertRaises(ValueError):
            ModelSettings(timeout_seconds=float("nan"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            path.write_text('[pipeline]\ntypo=1\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_settings(path)

    def test_metric_denominators(self):
        self.assertEqual(recall_at_k(["a", "a", "b"], ["a", "c"]), 0.5)
        self.assertEqual(candidate_retention(["a", "b"], ["a"], ["a", "c"]), 1)
        self.assertIsNone(candidate_retention(["b"], [], ["a"]))
        self.assertEqual(speedup(10, 5), 2)
        with self.assertRaises(ValueError):
            recall_at_k([], [])


class ClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_trace_records_transport_failure_without_credentials(self):
        trace = CallTrace()
        settings = {"extract": ModelSettings(model="test", max_attempts=1)}
        client = ChatClient(settings, 1, trace)
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(ModelError):
            await client.generate("extract", "系统", "输入")
        self.assertEqual(len(trace.records), 1)
        self.assertFalse(trace.records[0]["success"])
        self.assertGreaterEqual(trace.records[0]["duration_seconds"], 0)

    async def test_call_concurrency_is_bounded(self):
        trace = CallTrace()
        client = ChatClient({"extract": ModelSettings(model="test")}, 1, trace)
        with patch("literatureqa.client._send", return_value=("{}", {"total_tokens": 7})):
            values = await asyncio.gather(*(client.generate("extract", "系统", "输入") for _ in range(4)))
        self.assertEqual(values, ["{}"] * 4)
        self.assertEqual(len(trace.records), 4)
        self.assertTrue(all(row["success"] and row["usage"]["total_tokens"] == 7 for row in trace.records))


if __name__ == "__main__":
    unittest.main()
