"""网页与命令行共用的服务入口。"""

from .client import CallTrace, ChatClient
from .pipeline import QAPipeline
from .retrieval import CorpusRetriever
from .schema import load_corpus


def build_retriever(corpus_path, settings):
    return CorpusRetriever(load_corpus(corpus_path), settings.retrieval)


async def answer_question(question: str, retriever, settings) -> dict:
    trace = CallTrace()
    client = ChatClient(settings.models, settings.pipeline.model_concurrency, trace)
    pipeline = QAPipeline(retriever, client, settings.pipeline, trace)
    return await pipeline.ask(question)
