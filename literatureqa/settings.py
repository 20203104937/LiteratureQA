"""公共模型配置、四种角色覆盖配置与流程参数。"""

from dataclasses import dataclass, fields
import math
import os
from pathlib import Path
import tomllib

ROLES = ("decompose", "extract", "judge", "synthesize")


@dataclass(frozen=True)
class ModelSettings:
    base_url: str = ""
    model: str = ""
    api_key_env: str = "LITERATUREQA_API_KEY"
    timeout_seconds: float = 90
    max_attempts: int = 2
    max_output_tokens: int = 2048

    def __post_init__(self):
        if any(not isinstance(getattr(self, key), str) for key in ("base_url", "model", "api_key_env")):
            raise ValueError("模型地址、名称和密钥环境变量名必须是文字")
        if type(self.timeout_seconds) not in (int, float) or not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("接口超时时间必须为正数")
        for key in ("max_attempts", "max_output_tokens"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"{key} 必须是正整数")
        if self.max_attempts > 5:
            raise ValueError("单次模型请求最多尝试 5 次")


@dataclass(frozen=True)
class PipelineSettings:
    top_k: int = 10
    paper_concurrency: int = 8
    model_concurrency: int = 8
    max_subquestions: int = 6
    threshold_n: float = 0.5
    threshold_k: float = 0.5
    passage_count: int = 5
    passage_chars: int = 1600
    passage_overlap: int = 200
    max_abstract_chars: int = 4000
    max_final_evidence_chars: int = 30000

    def __post_init__(self):
        for key in ("top_k", "paper_concurrency", "model_concurrency", "max_subquestions", "passage_count", "passage_chars", "max_abstract_chars", "max_final_evidence_chars"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"{key} 必须是正整数")
        if type(self.passage_overlap) is not int or not 0 <= self.passage_overlap < self.passage_chars:
            raise ValueError("片段重叠长度必须小于片段长度")
        for key in ("threshold_n", "threshold_k"):
            value = getattr(self, key)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{key} 必须是非负有限数值")


@dataclass(frozen=True)
class RetrievalSettings:
    mode: str = "bm25"
    rrf_constant: int = 60
    embedding_model: str = "intfloat/multilingual-e5-small"
    device: str = "cpu"

    def __post_init__(self):
        if self.mode not in ("bm25", "semantic", "hybrid"):
            raise ValueError("检索方式只支持 bm25、semantic、hybrid")
        if type(self.rrf_constant) is not int or self.rrf_constant < 1:
            raise ValueError("排名融合常数必须是正整数")
        if any(not isinstance(getattr(self, key), str) or not getattr(self, key) for key in ("embedding_model", "device")):
            raise ValueError("嵌入模型名和计算设备必须是非空文字")


@dataclass(frozen=True)
class Settings:
    models: dict[str, ModelSettings]
    pipeline: PipelineSettings
    retrieval: RetrievalSettings


def _construct(kind, values):
    allowed = {item.name for item in fields(kind)}
    if not isinstance(values, dict) or set(values) - allowed:
        raise ValueError(f"{kind.__name__} 配置含有未知字段")
    return kind(**values)


def load_settings(path: str | Path | None = None) -> Settings:
    raw = {}
    if path is not None:
        with Path(path).open("rb") as stream:
            raw = tomllib.load(stream)
    if set(raw) - {"model", "roles", "pipeline", "retrieval"}:
        raise ValueError("配置含有未知章节")
    common = dict(raw.get("model", {}))
    for key in ("base_url", "model"):
        value = os.environ.get(f"LITERATUREQA_{key.upper()}")
        if value:
            common[key] = value
    overrides = raw.get("roles", {})
    if set(overrides) - set(ROLES):
        raise ValueError("存在未知模型角色")
    models = {role: _construct(ModelSettings, {**common, **overrides.get(role, {})}) for role in ROLES}
    return Settings(models, _construct(PipelineSettings, raw.get("pipeline", {})), _construct(RetrievalSettings, raw.get("retrieval", {})))
