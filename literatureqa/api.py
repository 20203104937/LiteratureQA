"""可选 FastAPI 服务；默认仅在本机启动。"""

import asyncio
from contextlib import asynccontextmanager
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .service import answer_question, build_retriever
from .settings import load_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = os.environ.get("LITERATUREQA_CONFIG")
    if config is None and Path("settings.local.toml").exists():
        config = "settings.local.toml"
    settings = load_settings(config)
    corpus = os.environ.get("LITERATUREQA_CORPUS", "data/papers.jsonl")
    app.state.settings = settings
    app.state.retriever = await asyncio.to_thread(build_retriever, corpus, settings)
    # 网页请求顺序处理，每个请求内部的论文与模型调用可以并发。
    app.state.request_gate = asyncio.Semaphore(1)
    yield


app = FastAPI(title="LiteratureQA", lifespan=lifespan)


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=10000)


@app.get("/health")
async def health():
    return {"status": "ok", "paper_count": len(app.state.retriever.papers)}


@app.post("/ask")
async def ask(request: Question):
    if not request.question.strip():
        raise HTTPException(status_code=422, detail="问题不能为空")
    async with app.state.request_gate:
        result = await answer_question(request.question, app.state.retriever, app.state.settings)
    # 处理失败仍返回完整诊断数据，客户端可查看 status 与 errors。
    return result
