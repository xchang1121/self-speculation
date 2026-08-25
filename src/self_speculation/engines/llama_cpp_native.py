"""In-process llama-cpp-python streaming and external draft integration."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import queue
import threading
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

from ..drafts import (
    BoundaryDraftStore,
    DraftReceipt,
    DraftRequest,
    DraftVerificationOutcome,
)
from ..models import (
    EngineCapabilities,
    InferenceRequest,
    StreamChunk,
    ToolCallDelta,
)


LlamaCppPromptRenderer = Callable[
    [InferenceRequest],
    str | Awaitable[str],
]
LlamaCppDraftRequestPredicate = Callable[[InferenceRequest], bool]


async def _resolve(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _numpy() -> Any:
    try:
        import numpy as np
    except ImportError as error:  # pragma: no cover - llama-cpp-python requires it
        raise ImportError(
            "llama.cpp draft injection requires: "
            "pip install 'self-speculation[llama-cpp]'"
        ) from error
    return np


class LlamaCppBoundaryDraftModel:
    """A llama-cpp-python ``LlamaDraftModel`` backed by boundary drafts.

    Pass this object to ``llama_cpp.Llama(..., draft_model=draft_model)`` at
    construction time. The high-level llama.cpp loop then batches and verifies
    returned candidates with the target model.
    """

    def __init__(
        self,
        store: BoundaryDraftStore,
        *,
        max_tokens: int = 28,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        self.store = store
        self.max_tokens = max_tokens
        self._local = threading.local()
        self.proposal_count = 0
        self.proposed_tokens = 0

    @property
    def active_request_id(self) -> str | None:
        return getattr(self._local, "request_id", None)

    @contextlib.contextmanager
    def bind(self, request_id: str):
        previous = self.active_request_id
        self._local.request_id = request_id
        try:
            yield self
        finally:
            if previous is None:
                try:
                    del self._local.request_id
                except AttributeError:
                    pass
            else:
                self._local.request_id = previous

    def __call__(self, input_ids: Any, /, **kwargs: Any) -> Any:
        del kwargs
        np = _numpy()
        request_id = self.active_request_id
        if request_id is None:
            return np.array([], dtype=np.intc)
        sequence = input_ids.tolist()
        if sequence and isinstance(sequence[0], list):
            raise ValueError("llama.cpp boundary drafts require one token sequence")
        proposal = self.store.offer(
            request_id,
            sequence,
            sequence_length=len(sequence),
            max_tokens=self.max_tokens,
        )
        if proposal is None:
            return np.array([], dtype=np.intc)
        self.proposal_count += 1
        self.proposed_tokens += len(proposal.token_ids)
        return np.asarray(proposal.token_ids, dtype=np.intc)


def _content(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(
            str(item.get("text", ""))
            for item in value
            if isinstance(item, Mapping)
        )
    return ""


def _tool_deltas(delta: Mapping[str, Any]) -> tuple[ToolCallDelta, ...]:
    result: list[ToolCallDelta] = []
    for item in delta.get("tool_calls") or ():
        if not isinstance(item, Mapping):
            continue
        function = item.get("function") or {}
        if not isinstance(function, Mapping):
            function = {}
        arguments = function.get("arguments", item.get("arguments", ""))
        result.append(
            ToolCallDelta(
                index=int(item.get("index", len(result))),
                call_id=item.get("id"),
                name=str(function.get("name") or item.get("name") or ""),
                arguments=str(arguments or ""),
            )
        )
    return tuple(result)


def _chunk(payload: Any) -> StreamChunk | None:
    if not isinstance(payload, Mapping):
        raise TypeError("llama-cpp-python stream items must be mappings")
    usage = {
        str(key): int(value)
        for key, value in (payload.get("usage") or {}).items()
        if isinstance(value, int)
    }
    choices = payload.get("choices") or ()
    if not choices:
        return StreamChunk(usage=usage, raw=payload) if usage else None
    choice = choices[0]
    if not isinstance(choice, Mapping):
        return None
    delta = choice.get("delta") or {}
    if not isinstance(delta, Mapping):
        delta = {}
    text = _content(delta.get("content")) if delta else _content(choice.get("text"))
    reasoning = ""
    for key in ("reasoning_content", "reasoning", "reasoning_text"):
        reasoning = _content(delta.get(key))
        if reasoning:
            break
    return StreamChunk(
        text=text,
        reasoning=reasoning,
        tool_call_deltas=_tool_deltas(delta),
        finish_reason=choice.get("finish_reason"),
        usage=usage,
        raw=payload,
    )


@dataclass(slots=True)
class _Failure:
    error: BaseException


class LlamaCppPythonEngine:
    """Adapt an in-process ``llama_cpp.Llama`` object.

    ``llama_cpp.Llama`` owns mutable target KV state, so calls on one instance
    are serialized. For a useful concurrent D1 fork, give ``ForkController`` a
    second engine backed by a separate ``Llama`` instance as ``fork_engine``.
    """

    def __init__(
        self,
        llama: Any,
        *,
        draft_model: LlamaCppBoundaryDraftModel | None = None,
        prompt_renderer: LlamaCppPromptRenderer | None = None,
        generation_kwargs: Mapping[str, Any] | None = None,
        prefix_cache: bool | None = False,
        draft_request_predicate: LlamaCppDraftRequestPredicate | None = None,
        name: str = "llama-cpp-python",
    ) -> None:
        if llama is None:
            raise ValueError("llama is required")
        installed_draft = getattr(llama, "draft_model", None)
        if draft_model is None and isinstance(
            installed_draft, LlamaCppBoundaryDraftModel
        ):
            draft_model = installed_draft
        if draft_model is not None and installed_draft is not draft_model:
            raise ValueError(
                "construct llama_cpp.Llama with draft_model set to the same "
                "LlamaCppBoundaryDraftModel; installing it afterwards does not "
                "enable the logits required for verification"
            )
        self.llama = llama
        self.draft_model = draft_model
        self.prompt_renderer = prompt_renderer
        self.generation_kwargs = dict(generation_kwargs or {})
        self.draft_request_predicate = draft_request_predicate or (
            lambda request: not request.request_id.endswith(":fork")
        )
        self.name = name
        self._generation_lock = threading.RLock()
        self.capabilities = EngineCapabilities(
            prompt=True,
            chat=True,
            structured_tool_deltas=True,
            prefix_cache=prefix_cache,
            draft_feedback=draft_model is not None,
        )

    async def render_prompt(self, request: InferenceRequest) -> str:
        if request.prompt is not None:
            return request.prompt
        if self.prompt_renderer is None:
            raise ValueError(
                "rendering a llama.cpp chat prompt requires prompt_renderer"
            )
        rendered = await _resolve(self.prompt_renderer(request))
        if not isinstance(rendered, str):
            raise TypeError("llama.cpp prompt_renderer must return a string")
        return rendered

    def tokenize_text(self, text: str) -> tuple[int, ...]:
        return tuple(
            int(token)
            for token in self.llama.tokenize(
                text.encode("utf-8"),
                add_bos=True,
                special=True,
            )
        )

    def tokenize_continuation(self, text: str) -> tuple[int, ...]:
        return tuple(
            int(token)
            for token in self.llama.tokenize(
                text.encode("utf-8"),
                add_bos=False,
                special=True,
            )
        )

    async def prompt_token_count(self, request: InferenceRequest) -> int:
        return len(self.tokenize_text(await self.render_prompt(request)))

    async def submit(self, draft: DraftRequest) -> DraftReceipt:
        if self.draft_model is None:
            raise RuntimeError("LlamaCppPythonEngine has no configured draft model")
        return self.draft_model.store.register(draft)

    async def clear(
        self, request_id: str
    ) -> DraftVerificationOutcome | None:
        if self.draft_model is None:
            return None
        return self.draft_model.store.take_outcome(request_id)

    def _options(self, request: InferenceRequest) -> dict[str, Any]:
        nested = request.extra.get("generate_kwargs") or {}
        if not isinstance(nested, Mapping):
            raise TypeError("extra['generate_kwargs'] must be a mapping")
        options = {**self.generation_kwargs, **dict(nested)}
        if request.max_tokens is not None:
            options.setdefault("max_tokens", request.max_tokens)
        if request.temperature is not None:
            options.setdefault("temperature", request.temperature)
        if request.stop:
            options.setdefault("stop", list(request.stop))
        options["stream"] = True
        return options

    def _source(self, request: InferenceRequest, options: dict[str, Any]) -> Any:
        if request.prompt is not None:
            return self.llama.create_completion(
                prompt=request.prompt,
                **options,
            )
        if self.prompt_renderer is not None:
            raise RuntimeError("rendered chat prompts are resolved before _source")
        chat_options = dict(options)
        if request.tools:
            chat_options.setdefault("tools", [dict(tool) for tool in request.tools])
        return self.llama.create_chat_completion(
            messages=[dict(message) for message in request.messages],
            **chat_options,
        )

    async def stream(self, request: InferenceRequest) -> AsyncIterator[StreamChunk]:
        effective_request = request
        if request.messages and self.prompt_renderer is not None:
            effective_request = request.with_changes(
                prompt=await self.render_prompt(request),
                messages=(),
            )
        options = self._options(effective_request)
        items: queue.Queue[Any] = queue.Queue()
        stop_event = threading.Event()
        end = object()
        use_draft = (
            self.draft_model is not None
            and self.draft_request_predicate(effective_request)
        )

        def produce() -> None:
            binding = (
                self.draft_model.bind(effective_request.request_id)
                if use_draft and self.draft_model is not None
                else contextlib.nullcontext()
            )
            source = None
            try:
                with self._generation_lock, binding:
                    source = self._source(effective_request, options)
                    for payload in source:
                        if stop_event.is_set():
                            break
                        items.put(payload)
            except BaseException as error:
                items.put(_Failure(error))
            finally:
                close = getattr(source, "close", None)
                if close is not None:
                    close()
                items.put(end)

        producer = asyncio.create_task(asyncio.to_thread(produce))
        completed = False
        try:
            while True:
                item = await asyncio.to_thread(items.get)
                if item is end:
                    break
                if isinstance(item, _Failure):
                    raise item.error
                chunk = _chunk(item)
                if chunk is not None:
                    yield chunk
            await producer
            completed = True
        finally:
            stop_event.set()
            if not completed:
                try:
                    await producer
                except BaseException:
                    pass
