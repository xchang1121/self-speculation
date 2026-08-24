"""OpenAI-compatible SSE adapter for vLLM, SGLang, TGI, and llama.cpp."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping
from typing import Any, TYPE_CHECKING

from ..models import (
    EngineCapabilities,
    InferenceRequest,
    StreamChunk,
    TokenLogprob,
    ToolCallDelta,
)

if TYPE_CHECKING:
    import httpx


class OpenAIStreamError(RuntimeError):
    pass


def _httpx() -> Any:
    try:
        import httpx
    except ImportError as error:  # pragma: no cover - exercised without the extra
        raise ImportError(
            "OpenAICompatibleEngine requires: pip install 'self-speculation[http]'"
        ) from error
    return httpx


def _content_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for part in value:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, Mapping):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


def _argument_fragment(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _tool_deltas(delta: Mapping[str, Any]) -> tuple[ToolCallDelta, ...]:
    calls: list[ToolCallDelta] = []
    for item in delta.get("tool_calls") or ():
        if not isinstance(item, Mapping):
            continue
        function = item.get("function") or {}
        if not isinstance(function, Mapping):
            function = {}
        calls.append(
            ToolCallDelta(
                index=int(item.get("index", len(calls))),
                call_id=item.get("id"),
                name=str(function.get("name") or item.get("name") or ""),
                arguments=_argument_fragment(
                    function.get("arguments", item.get("arguments"))
                ),
            )
        )
    return tuple(calls)


def _completion_logprobs(value: Any) -> tuple[TokenLogprob, ...]:
    if not isinstance(value, Mapping):
        return ()
    tokens = list(value.get("tokens") or ())
    token_logprobs = list(value.get("token_logprobs") or ())
    top_logprobs = list(value.get("top_logprobs") or ())
    result: list[TokenLogprob] = []
    for index, token in enumerate(tokens):
        logprob = token_logprobs[index] if index < len(token_logprobs) else None
        top = top_logprobs[index] if index < len(top_logprobs) else {}
        result.append(
            TokenLogprob(
                token=str(token),
                logprob=logprob,
                top_logprobs=dict(top) if isinstance(top, Mapping) else {},
            )
        )
    return tuple(result)


def _chat_logprobs(value: Any) -> tuple[TokenLogprob, ...]:
    if not isinstance(value, Mapping):
        return ()
    result: list[TokenLogprob] = []
    for item in value.get("content") or ():
        if not isinstance(item, Mapping):
            continue
        top = {
            str(alternative.get("token", "")): alternative.get("logprob")
            for alternative in item.get("top_logprobs") or ()
            if isinstance(alternative, Mapping)
            and alternative.get("logprob") is not None
        }
        result.append(
            TokenLogprob(
                token=str(item.get("token", "")),
                logprob=item.get("logprob"),
                top_logprobs=top,
            )
        )
    return tuple(result)


def _usage(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): item for key, item in value.items() if isinstance(item, int)}


class OpenAICompatibleEngine:
    """Stream either raw completions or chat completions from one endpoint."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = 300.0,
        client: "httpx.AsyncClient | None" = None,
        name: str = "openai-compatible",
        choice_index: int = 0,
        prefix_cache: bool | None = None,
    ) -> None:
        if not base_url.strip():
            raise ValueError("base_url must not be empty")
        if choice_index < 0:
            raise ValueError("choice_index must be non-negative")
        module = _httpx()
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.headers = dict(headers or {})
        self.timeout = timeout
        self.name = name
        self.choice_index = choice_index
        self.capabilities = EngineCapabilities(
            prompt=True,
            chat=True,
            structured_tool_deltas=True,
            token_ids=True,
            logprobs=True,
            prefix_cache=prefix_cache,
        )
        self._client = client or module.AsyncClient()
        self._owns_client = client is None

    @property
    def completions_url(self) -> str:
        return self.base_url + "/completions"

    @property
    def chat_completions_url(self) -> str:
        return self.base_url + "/chat/completions"

    def _headers(self) -> dict[str, str]:
        headers = dict(self.headers)
        if self.api_key and "Authorization" not in headers:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _payload(self, request: InferenceRequest) -> tuple[str, dict[str, Any]]:
        body = dict(request.extra)
        body["stream"] = True
        if request.model is not None:
            body["model"] = request.model
        if request.max_tokens is not None:
            body["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.stop:
            body["stop"] = list(request.stop)

        if request.prompt is not None:
            body["prompt"] = request.prompt
            return self.completions_url, body

        body["messages"] = [dict(message) for message in request.messages]
        if request.tools:
            body["tools"] = [dict(tool) for tool in request.tools]
        return self.chat_completions_url, body

    def _chunk(self, payload: Mapping[str, Any]) -> StreamChunk | None:
        choice = next(
            (
                item
                for item in payload.get("choices") or ()
                if isinstance(item, Mapping)
                and int(item.get("index", 0)) == self.choice_index
            ),
            None,
        )
        usage = _usage(payload.get("usage"))
        if choice is None:
            return StreamChunk(usage=usage, raw=payload) if usage else None

        delta = choice.get("delta") or choice.get("message") or {}
        if not isinstance(delta, Mapping):
            delta = {}
        is_chat = bool(delta) or "delta" in choice or "message" in choice
        text = (
            _content_text(delta.get("content"))
            if is_chat
            else _content_text(choice.get("text"))
        )
        reasoning = ""
        if is_chat:
            for key in ("reasoning_content", "reasoning", "reasoning_text"):
                reasoning = _content_text(delta.get(key))
                if reasoning:
                    break
        raw_token_ids = choice.get("token_ids") or payload.get("token_ids") or ()
        token_ids = tuple(int(item) for item in raw_token_ids)
        return StreamChunk(
            text=text,
            reasoning=reasoning,
            tool_call_deltas=_tool_deltas(delta) if is_chat else (),
            token_ids=token_ids,
            logprobs=(
                _chat_logprobs(choice.get("logprobs"))
                if is_chat
                else _completion_logprobs(choice.get("logprobs"))
            ),
            finish_reason=choice.get("finish_reason"),
            usage=usage,
            raw=payload,
        )

    async def stream(self, request: InferenceRequest) -> AsyncIterator[StreamChunk]:
        url, body = self._payload(request)
        async with self._client.stream(
            "POST",
            url,
            json=body,
            headers=self._headers(),
            timeout=self.timeout,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                stripped = line.strip()
                if not stripped or stripped.startswith(":") or stripped.startswith("event:"):
                    continue
                data = stripped[5:].lstrip() if stripped.startswith("data:") else stripped
                if data == "[DONE]":
                    break
                try:
                    payload = json.loads(data)
                except json.JSONDecodeError as error:
                    raise OpenAIStreamError(f"invalid SSE JSON: {data[:200]}") from error
                if not isinstance(payload, Mapping):
                    continue
                if payload.get("error"):
                    raise OpenAIStreamError(str(payload["error"]))
                chunk = self._chunk(payload)
                if chunk is not None:
                    yield chunk

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> "OpenAICompatibleEngine":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.aclose()


class VLLMEngine(OpenAICompatibleEngine):
    def __init__(self, base_url: str, **kwargs: Any) -> None:
        kwargs.setdefault("name", "vllm")
        super().__init__(base_url, **kwargs)


class SGLangEngine(OpenAICompatibleEngine):
    def __init__(self, base_url: str, **kwargs: Any) -> None:
        kwargs.setdefault("name", "sglang")
        super().__init__(base_url, **kwargs)


class TGIEngine(OpenAICompatibleEngine):
    def __init__(self, base_url: str, **kwargs: Any) -> None:
        kwargs.setdefault("name", "tgi")
        super().__init__(base_url, **kwargs)


class LlamaCppEngine(OpenAICompatibleEngine):
    def __init__(self, base_url: str, **kwargs: Any) -> None:
        kwargs.setdefault("name", "llama.cpp")
        super().__init__(base_url, **kwargs)
