"""关键词检索、可选语义检索及倒数排名融合。"""

from collections import Counter
import math
import re
import threading

from .schema import Paper, Passage
from .settings import RetrievalSettings


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9_]+", text.lower())
    for run in re.findall(r"[\u3400-\u9fff]+", text):
        tokens.extend(run)
        tokens.extend(run[i:i + 2] for i in range(len(run) - 1))
    return tokens


class KeywordIndex:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.counts = [Counter(tokenize(text)) for text in texts]
        self.lengths = [sum(count.values()) for count in self.counts]
        self.average = sum(self.lengths) / len(texts) if texts else 1
        self.average = self.average or 1
        self.frequency = Counter(term for count in self.counts for term in count)
        self.k1, self.b = k1, b

    def rank(self, query: str) -> list[tuple[int, float]]:
        scores = []
        total = len(self.counts)
        for index, counts in enumerate(self.counts):
            score = 0.0
            for term in set(tokenize(query)):
                count = counts.get(term, 0)
                if not count:
                    continue
                frequency = self.frequency[term]
                rarity = math.log(1 + (total - frequency + 0.5) / (frequency + 0.5))
                normalizer = self.k1 * (1 - self.b + self.b * self.lengths[index] / self.average)
                score += rarity * count * (self.k1 + 1) / (count + normalizer)
            if score > 0:
                scores.append((index, score))
        return sorted(scores, key=lambda item: (-item[1], item[0]))


def fuse_rankings(rankings: list[list[tuple[int, float]]], constant: int) -> list[tuple[int, float]]:
    if constant <= 0:
        raise ValueError("融合常数必须为正数")
    fused = Counter()
    for ranking in rankings:
        seen = set()
        for position, (index, _) in enumerate(ranking, 1):
            if index in seen:
                raise ValueError("单路排名包含重复文档")
            seen.add(index)
            fused[index] += 1 / (constant + position)
    return sorted(fused.items(), key=lambda item: (-item[1], item[0]))


class SemanticEncoder:
    def __init__(self, settings: RetrievalSettings):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(settings.embedding_model, device=settings.device)
        self.lock = threading.Lock()
        self.use_e5_prefix = "e5" in settings.embedding_model.lower()

    def encode(self, texts: list[str], query: bool = False):
        prefix = ("query: " if query else "passage: ") if self.use_e5_prefix else ""
        with self.lock:
            return self.model.encode([prefix + text for text in texts], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)


def make_passages(text: str, size: int, overlap: int) -> list[Passage]:
    if size < 1 or not 0 <= overlap < size:
        raise ValueError("片段长度或重叠长度不合法")
    result = []
    for start in range(0, len(text), size - overlap):
        chunk = text[start:start + size]
        if chunk.strip():
            result.append(Passage(start, chunk))
        if start + size >= len(text):
            break
    return result


class CorpusRetriever:
    def __init__(self, papers: list[Paper], settings: RetrievalSettings, encoder=None):
        self.papers = tuple(papers)
        self.settings = settings
        self.encoder = encoder
        if self.encoder is None and settings.mode != "bm25":
            self.encoder = SemanticEncoder(settings)
        self.texts = [paper.title + "\n" + paper.abstract for paper in papers]
        self.keyword = KeywordIndex(self.texts)
        self.vectors = self.encoder.encode(self.texts) if self.encoder else None

    def _rank(self, query: str, texts: list[str], keyword: KeywordIndex, vectors=None):
        rankings = []
        if self.settings.mode in ("bm25", "hybrid"):
            rankings.append(keyword.rank(query))
        if self.settings.mode in ("semantic", "hybrid"):
            if not texts:
                return []
            vectors = vectors if vectors is not None else self.encoder.encode(texts)
            query_vector = self.encoder.encode([query], query=True)[0]
            scores = vectors @ query_vector
            rankings.append(sorted(enumerate(scores.tolist()), key=lambda item: (-item[1], item[0])))
        return fuse_rankings(rankings, self.settings.rrf_constant) if self.settings.mode == "hybrid" else rankings[0]

    def search(self, query: str, limit: int) -> list[Paper]:
        return [self.papers[index] for index, _ in self._rank(query, self.texts, self.keyword, self.vectors)[:limit]]

    def passages(self, paper: Paper, query: str, limit: int, size: int, overlap: int) -> list[Passage]:
        chunks = make_passages(paper.full_text, size, overlap)
        texts = [chunk.text for chunk in chunks]
        ranking = self._rank(query, texts, KeywordIndex(texts))
        return [chunks[index] for index, _ in ranking[:limit]]
