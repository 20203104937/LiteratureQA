"""离线演示响应，仅用于查看数据如何流转，不代表真实模型能力。"""

import json
import re

from .client import CallTrace
from .pipeline import QAPipeline


class DemoClient:
    async def generate(self, role: str, system: str, user: str) -> str:
        payload = json.loads(user)
        if role == "decompose":
            value = {"questions": [payload["question"]]}
        elif role == "extract":
            text = payload["documents"][0]["text"]
            sentences = [part.strip() for part in re.split(r"(?<=[。.!?])\s*", text) if part.strip()]
            quote = sentences[0] if sentences else text
            value = {"answer": quote, "quotes": [quote]}
        elif role == "judge":
            value = {"score": 80}
        else:
            value = {"claims": [{"text": row["answer"], "citations": [row["number"]]} for row in payload["evidence"][:3]]}
        return json.dumps(value, ensure_ascii=False)


async def demo_question(question: str, retriever, settings) -> dict:
    result = await QAPipeline(retriever, DemoClient(), settings.pipeline, CallTrace()).ask(question)
    result["mode"] = "offline_demo"
    result["notice"] = "演示使用固定规则和固定 80 分，未调用模型；答案、评分与耗时不能作为模型评测结果。"
    return result
