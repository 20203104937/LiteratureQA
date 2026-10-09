"""通过标准库调用兼容聊天接口；记录真实请求耗时与失败。"""

import asyncio
from dataclasses import dataclass, field
import json
import os
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from .settings import ModelSettings


class ModelError(RuntimeError):
    pass


@dataclass
class CallTrace:
    records: list[dict] = field(default_factory=list)

    def record(self, **values):
        self.records.append(values)


def _endpoint(base_url: str) -> str:
    parsed = urlsplit(base_url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ModelError("请配置合法的模型服务地址，地址中不要包含凭证或查询参数")
    return base_url.rstrip("/") + "/chat/completions"


def _send(settings: ModelSettings, system: str, user: str) -> tuple[str, dict]:
    if not settings.model:
        raise ModelError("请填写模型名称")
    key = os.environ.get(settings.api_key_env, "")
    if not key:
        raise ModelError(f"请设置密钥环境变量 {settings.api_key_env}")
    body = json.dumps({"model": settings.model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}], "temperature": 0, "max_tokens": settings.max_output_tokens}, ensure_ascii=False).encode("utf-8")
    request = Request(_endpoint(settings.base_url), data=body, headers={"Content-Type": "application/json", "Authorization": "Bearer " + key}, method="POST")
    with urlopen(request, timeout=settings.timeout_seconds) as response:
        raw = response.read(8_000_001)
    if len(raw) > 8_000_000:
        raise ModelError("模型响应超过大小限制")
    try:
        data = json.loads(raw)
        choice = data["choices"][0]
        content = choice["message"]["content"]
        if not isinstance(content, str) or not content.strip() or choice.get("finish_reason") == "length":
            raise ValueError("空响应或输出被截断")
    except (TypeError, KeyError, IndexError, ValueError) as exc:
        raise ModelError("模型响应为空、截断或不符合聊天接口格式") from exc
    usage = data.get("usage", {})
    usage = {key: value for key, value in usage.items() if key in ("prompt_tokens", "completion_tokens", "total_tokens") and type(value) is int} if isinstance(usage, dict) else {}
    return content, usage


class ChatClient:
    def __init__(self, models: dict[str, ModelSettings], concurrency: int, trace: CallTrace):
        self.models = models
        self.gate = asyncio.Semaphore(concurrency)
        self.trace = trace

    async def generate(self, role: str, system: str, user: str) -> str:
        settings = self.models[role]
        for attempt in range(1, settings.max_attempts + 1):
            queued = time.perf_counter()
            async with self.gate:
                started = time.perf_counter()
                success = False
                usage = {}
                retry = False
                error = ""
                try:
                    content, usage = await asyncio.to_thread(_send, settings, system, user)
                    success = True
                    return content
                except HTTPError as exc:
                    error = f"模型接口 HTTP {exc.code}"
                    retry = exc.code in (408, 429, 500, 502, 503, 504)
                except (URLError, TimeoutError, OSError):
                    error = "模型接口网络失败或超时"
                    retry = True
                except ModelError as exc:
                    error = str(exc)
                finally:
                    self.trace.record(role=role, model=settings.model, attempt=attempt, success=success, error=error, queue_seconds=started - queued, duration_seconds=time.perf_counter() - started, usage=usage)
            if not retry or attempt == settings.max_attempts:
                raise ModelError(error)
            await asyncio.sleep(min(2 ** (attempt - 1), 8))
        raise ModelError("模型请求失败")
