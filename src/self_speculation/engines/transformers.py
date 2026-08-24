"""In-process Hugging Face Transformers streaming adapter."""

from __future__ import annotations

import asyncio
import inspect
import threading
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from typing import Any

from ..models import EngineCapabilities, InferenceRequest, StreamChunk


TransformersPromptRenderer = Callable[
    [InferenceRequest],
    str | Awaitable[str],
]


async def _resolve(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _transformers() -> tuple[Any, Any, Any]:
    try:
        from transformers import StoppingCriteria, StoppingCriteriaList
        from transformers import TextIteratorStreamer
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ImportError(
            "TransformersEngine requires: "
            "pip install 'self-speculation[transformers]'"
        ) from error
    return (StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer)


def _flatten_token_ids(value: Any) -> tuple[int, ...]:
    raw = value.tolist()
    if raw and isinstance(raw[0], list):
        if len(raw) != 1:
            raise ValueError("TransformersEngine only supports batch size 1")
        raw = raw[0]
    return tuple(int(token_id) for token_id in raw)


def _make_streamer(
    tokenizer: Any,
    *,
    skip_special_tokens: bool,
    clean_up_tokenization_spaces: bool,
) -> Any:
    _, _, TextIteratorStreamer = _transformers()

    class ChunkStreamer(TextIteratorStreamer):
        def __init__(self) -> None:
            super().__init__(
                tokenizer,
                skip_prompt=True,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            )
            self._pending_token_ids: list[int] = []

        def put(self, value: Any) -> None:
            is_prompt = self.skip_prompt and self.next_tokens_are_prompt
            if not is_prompt:
                self._pending_token_ids.extend(_flatten_token_ids(value))
            super().put(value)

        def on_finalized_text(
            self,
            text: str,
            stream_end: bool = False,
        ) -> None:
            if text or (stream_end and self._pending_token_ids):
                self.text_queue.put(
                    StreamChunk(
                        text=text,
                        token_ids=tuple(self._pending_token_ids),
                    ),
                    timeout=self.timeout,
                )
                self._pending_token_ids.clear()
            if stream_end:
                self.text_queue.put(self.stop_signal, timeout=self.timeout)

    return ChunkStreamer()


def _shape_length(value: Any) -> int:
    shape = getattr(value, "shape", ())
    if not shape:
        raise TypeError("tokenizer input_ids must expose a shape")
    return int(shape[-1])


def _to_device(inputs: Any, device: Any) -> Any:
    if device is None:
        return inputs
    move = getattr(inputs, "to", None)
    if move is not None:
        return move(device)
    if not isinstance(inputs, Mapping):
        raise TypeError("tokenizer output must be a mapping or expose to(device)")
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }


class TransformersEngine:
    """Stream an in-process Transformers ``generate`` call.

    Generation runs in a worker thread so the blocking PyTorch loop does not
    stall the controller event loop. The adapter supports one sequence per
    request, matching Transformers' streaming and assisted-generation limits.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        prompt_renderer: TransformersPromptRenderer | None = None,
        generation_kwargs: Mapping[str, Any] | None = None,
        add_special_tokens: bool = True,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = False,
        name: str = "transformers",
    ) -> None:
        if model is None:
            raise ValueError("model is required")
        if tokenizer is None:
            raise ValueError("tokenizer is required")
        self.model = model
        self.tokenizer = tokenizer
        self.prompt_renderer = prompt_renderer
        self.generation_kwargs = dict(generation_kwargs or {})
        self.add_special_tokens = add_special_tokens
        self.skip_special_tokens = skip_special_tokens
        self.clean_up_tokenization_spaces = clean_up_tokenization_spaces
        self.name = name
        self.capabilities = EngineCapabilities(
            prompt=True,
            chat=True,
            token_ids=True,
            prefix_cache=False,
        )

    async def render_prompt(self, request: InferenceRequest) -> str:
        if request.prompt is not None:
            return request.prompt
        if self.prompt_renderer is not None:
            rendered = await _resolve(self.prompt_renderer(request))
        else:
            apply_template = getattr(self.tokenizer, "apply_chat_template", None)
            if apply_template is None:
                raise ValueError(
                    "chat requests require tokenizer.apply_chat_template or "
                    "prompt_renderer"
                )
            options: dict[str, Any] = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            if request.tools:
                options["tools"] = [dict(tool) for tool in request.tools]
            rendered = apply_template(
                [dict(message) for message in request.messages],
                **options,
            )
        if not isinstance(rendered, str):
            raise TypeError("Transformers prompt renderer must return a string")
        return rendered

    def tokenize_text(self, text: str) -> tuple[int, ...]:
        encode = getattr(self.tokenizer, "encode", None)
        if encode is None:
            encoded = self.tokenizer(
                text,
                add_special_tokens=self.add_special_tokens,
            )["input_ids"]
        else:
            encoded = encode(
                text,
                add_special_tokens=self.add_special_tokens,
            )
        if hasattr(encoded, "tolist"):
            encoded = encoded.tolist()
        if encoded and isinstance(encoded[0], list):
            if len(encoded) != 1:
                raise ValueError("TransformersEngine only supports batch size 1")
            encoded = encoded[0]
        return tuple(int(token_id) for token_id in encoded)

    async def prompt_token_count(self, request: InferenceRequest) -> int:
        return len(self.tokenize_text(await self.render_prompt(request)))

    def _request_generation_kwargs(
        self,
        request: InferenceRequest,
    ) -> dict[str, Any]:
        nested = request.extra.get("generate_kwargs") or {}
        if not isinstance(nested, Mapping):
            raise TypeError("extra['generate_kwargs'] must be a mapping")
        options = {**self.generation_kwargs, **dict(nested)}
        if request.max_tokens is not None:
            options.setdefault("max_new_tokens", request.max_tokens)
        if request.temperature is not None:
            if request.temperature <= 0:
                options.setdefault("do_sample", False)
                options.pop("temperature", None)
            else:
                options.setdefault("do_sample", True)
                options.setdefault("temperature", request.temperature)
        if request.stop:
            options.setdefault("stop_strings", list(request.stop))
            options.setdefault("tokenizer", self.tokenizer)
        return options

    async def _inputs(self, prompt: str) -> Any:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=self.add_special_tokens,
        )
        if not isinstance(inputs, Mapping) and not hasattr(inputs, "to"):
            raise TypeError("tokenizer must return a mapping or BatchEncoding")
        return _to_device(inputs, getattr(self.model, "device", None))

    async def stream(self, request: InferenceRequest) -> AsyncIterator[StreamChunk]:
        StoppingCriteria, StoppingCriteriaList, _ = _transformers()
        prompt = await self.render_prompt(request)
        inputs = await self._inputs(prompt)
        input_ids = inputs["input_ids"]
        prompt_tokens = _shape_length(input_ids)
        stop_event = threading.Event()

        class StopOnCancellation(StoppingCriteria):
            def __call__(self, *args: Any, **kwargs: Any) -> bool:
                del args, kwargs
                return stop_event.is_set()

        options = self._request_generation_kwargs(request)
        supplied_criteria = options.pop("stopping_criteria", None)
        criteria = list(supplied_criteria or ())
        criteria.append(StopOnCancellation())
        options["stopping_criteria"] = StoppingCriteriaList(criteria)
        streamer = _make_streamer(
            self.tokenizer,
            skip_special_tokens=self.skip_special_tokens,
            clean_up_tokenization_spaces=self.clean_up_tokenization_spaces,
        )
        options["streamer"] = streamer

        def generate() -> Any:
            try:
                return self.model.generate(**dict(inputs), **options)
            except BaseException:
                streamer.end()
                raise

        generation_task = asyncio.create_task(asyncio.to_thread(generate))
        iterator = iter(streamer)
        completed = False
        try:
            while True:
                chunk = await asyncio.to_thread(next, iterator, None)
                if chunk is None:
                    break
                if not isinstance(chunk, StreamChunk):
                    raise TypeError("Transformers streamer produced an invalid chunk")
                yield chunk
            output = await generation_task
            completed = True
            sequences = getattr(output, "sequences", output)
            total_tokens = _shape_length(sequences)
            completion_tokens = max(0, total_tokens - prompt_tokens)
            finish_reason = (
                "length"
                if request.max_tokens is not None
                and completion_tokens >= request.max_tokens
                else "stop"
            )
            yield StreamChunk(
                finish_reason=finish_reason,
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                },
                raw=output,
            )
        finally:
            stop_event.set()
            if not completed:
                try:
                    await generation_task
                except BaseException:
                    pass
