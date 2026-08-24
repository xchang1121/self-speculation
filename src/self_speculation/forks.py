"""Composable request builders for a speculative inference fork."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Mapping, Protocol, runtime_checkable

from .models import InferenceRequest, StreamSnapshot


PromptRenderer = Callable[
    [InferenceRequest],
    str | Awaitable[str],
]
RequestFactory = Callable[
    [InferenceRequest, StreamSnapshot],
    InferenceRequest | Awaitable[InferenceRequest],
]


class ForkBuildError(ValueError):
    """Raised when a fork request cannot be derived from the main request."""


@runtime_checkable
class ForkRequestBuilder(Protocol):
    name: str

    async def build(
        self,
        main_request: InferenceRequest,
        snapshot: StreamSnapshot,
    ) -> InferenceRequest:
        ...


async def _resolve(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


@dataclass(slots=True)
class PrefixForkBuilder:
    """Build a raw-completion fork by appending observed output and a prefix.

    For raw main requests, the original prompt is reused automatically. Chat
    requests require either ``base_prompt`` or a ``prompt_renderer`` that
    renders the same chat template used by the serving engine. Launching after
    the first main chunk lets engines with prefix caching reuse the populated
    prefix, as in SPORK D1.
    """

    forced_prefix: str
    name: str = "prefix"
    base_prompt: str | None = None
    prompt_renderer: PromptRenderer | None = None
    include_observed: bool = True
    max_tokens: int = 256
    temperature: float | None = 0.0
    stop: tuple[str, ...] = ()
    extra: Mapping[str, Any] = field(default_factory=dict)
    request_id_suffix: str = ":fork"

    def __post_init__(self) -> None:
        self.stop = tuple(self.stop)
        if not self.forced_prefix:
            raise ValueError("forced_prefix must not be empty")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.base_prompt is not None and self.prompt_renderer is not None:
            raise ValueError("base_prompt and prompt_renderer are mutually exclusive")

    async def _base(self, main_request: InferenceRequest) -> str:
        if self.base_prompt is not None:
            return self.base_prompt
        if self.prompt_renderer is not None:
            rendered = await _resolve(self.prompt_renderer(main_request))
            if not isinstance(rendered, str):
                raise ForkBuildError("prompt_renderer must return a string")
            return rendered
        if main_request.prompt is not None:
            return main_request.prompt
        raise ForkBuildError(
            "chat requests need base_prompt or prompt_renderer for a prefix fork"
        )

    async def build(
        self,
        main_request: InferenceRequest,
        snapshot: StreamSnapshot,
    ) -> InferenceRequest:
        base = await self._base(main_request)
        observed = snapshot.generated_text if self.include_observed else ""
        return InferenceRequest(
            prompt=base + observed + self.forced_prefix,
            model=main_request.model,
            request_id=main_request.request_id + self.request_id_suffix,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stop=self.stop,
            extra=dict(self.extra),
        )


@dataclass(slots=True)
class CallableForkBuilder:
    """Adapt an arbitrary sync or async request factory."""

    factory: RequestFactory
    name: str = "callable"

    async def build(
        self,
        main_request: InferenceRequest,
        snapshot: StreamSnapshot,
    ) -> InferenceRequest:
        request = await _resolve(self.factory(main_request, snapshot))
        if not isinstance(request, InferenceRequest):
            raise ForkBuildError("fork request factory must return InferenceRequest")
        return request
