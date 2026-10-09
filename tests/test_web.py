"""网页依赖未安装时跳过；所有模型响应均模拟，不发送付费请求。"""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from literatureqa.ingest import ingest_markdown

try:
    from fastapi.testclient import TestClient
    from literatureqa.api import app
except ImportError:
    TestClient = None


def fake_send(settings, system, user):
    payload = json.loads(user)
    if "evidence" in payload:
        value = {"claims": [{"text": "文献讨论了检索方法。", "citations": [1]}]}
    elif "answer" in payload:
        value = {"score": 80}
    elif "documents" in payload:
        text = payload["documents"][0]["text"]
        value = {"answer": "文献讨论了检索方法。", "quotes": [text]}
    else:
        value = {"questions": [payload["question"]]}
    return json.dumps(value, ensure_ascii=False), {"total_tokens": 10}


@unittest.skipIf(TestClient is None, "未安装可选网页测试依赖")
class WebTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        self.corpus = root / "papers.jsonl"
        examples = Path(__file__).resolve().parents[1] / "examples" / "papers"
        ingest_markdown(examples, self.corpus)
        self.environment = patch.dict(os.environ, {"LITERATUREQA_CORPUS": str(self.corpus)}, clear=True)
        self.environment.start()

    def tearDown(self):
        self.environment.stop()
        self.directory.cleanup()

    def test_health(self):
        with TestClient(app) as client:
            result = client.get("/health")
            self.assertEqual(result.status_code, 200)
            self.assertEqual(result.json(), {"status": "ok", "paper_count": 2})

    def test_empty_question_rejected(self):
        with TestClient(app) as client:
            self.assertEqual(client.post("/ask", json={"question": "   "}).status_code, 422)
            self.assertEqual(client.post("/ask", json={"question": ""}).status_code, 422)

    def test_full_api_with_fake_transport(self):
        with TestClient(app) as client, patch("literatureqa.client._send", side_effect=fake_send):
            first = client.post("/ask", json={"question": "关键词检索"}).json()
            second = client.post("/ask", json={"question": "语义检索"}).json()
        self.assertEqual(first["status"], "ok")
        self.assertEqual(second["status"], "ok")
        self.assertTrue(first["references"])
        self.assertNotEqual(first["request_id"], second["request_id"])
        self.assertEqual(len(first["calls"]), 6)
        self.assertEqual(len(second["calls"]), 6)


@unittest.skipIf(TestClient is None, "未安装可选网页测试依赖")
class UITests(unittest.TestCase):
    def test_page_loads_without_backend_request(self):
        from streamlit.testing.v1 import AppTest
        path = Path(__file__).resolve().parents[1] / "ui.py"
        page = AppTest.from_file(str(path)).run(timeout=15)
        self.assertFalse(page.exception)
        self.assertEqual(page.title[0].value, "LiteratureQA 文献问答")

    def test_answer_and_reference_display(self):
        from streamlit.testing.v1 import AppTest
        path = Path(__file__).resolve().parents[1] / "ui.py"
        class Response:
            def raise_for_status(self):
                pass
            def json(self):
                return {"answer": "引用示例 [1]", "references": [{"number": 1, "title": "示例论文", "quote": "原文", "source": "example.md", "start": 0, "end": 2, "phase": "abstract"}], "errors": []}
        page = AppTest.from_file(str(path)).run(timeout=15)
        page.text_area[0].set_value("关键词")
        with patch("requests.post", return_value=Response()) as post:
            page.button[0].click().run(timeout=15)
        self.assertFalse(page.exception)
        self.assertTrue(post.called)
        self.assertEqual(page.expander[0].label, "[1] 示例论文")
