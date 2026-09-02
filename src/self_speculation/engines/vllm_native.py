"""Duck-typed adapter for vLLM AsyncLLM and AsyncLLMEngine."""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterator, Mapping
from typing import Any, Awaitable, Callable, Literal

from ..models import EngineCapabilities, InferenceRequest, StreamChunk, TokenLogprob


SamplingParamsFactory = Callable[[InferenceRequest], Any | Awaitable[Any]]
NativePromptRenderer = Callable[[InferenceRequest], Any | Awaitable[Any]]
NativePromptTokenCounter = Callable[[InferenceRequest], int | Awaitable[int]]
OutputMode = Literal["delta", "cumulative", "auto"]


async def _resolve(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _native_logprobs(value: Any) -> tuple[TokenLogprob, ...]:
    if not isinstance(value, list):
        return ()
    result: list[TokenLogprob] = []
    for position in value:
        if not isinstance(position, Mapping) or not position:
            continue
        alternatives: dict[str, float] = {}
        selected_token = ""
        selected_logprob: float | None = None
        for token_id, item in position.items():
            token = getattr(item, "decoded_token", None)
            token_text = str(token if token is not None else token_id)
            logprob = getattr(item, "logprob", item if isinstance(item, float) else None)
            if logprob is not None:
                alternatives[token_text] = float(logprob)
            rank = getattr(item, "rank", None)
            if selected_logprob is None or rank == 1:
                selected_token = token_text
                selected_logprob = float(logprob) if logprob is not None else None
        result.append(TokenLogprob(selected_token, selected_logprob, alternatives))
    return tuple(result)


class VLLMNativeEngine:
    """Normalize vLLM's in-process request outputs into stream chunks."""

    def __init__(
        self,
        engine: Any,
        *,
        sampling_params_factory: SamplingParamsFactory | None = None,
        prompt_renderer: NativePromptRenderer | None = None,
        output_mode: OutputMode = "auto",
        choice_index: int = 0,
        prefix_cache: bool | None = None,
        max_context_tokens: int | None = None,
        prompt_token_counter: NativePromptTokenCounter | None = None,
        name: str = "vllm-native",
    ) -> None:
        if output_mode not in ("delta", "cumulative", "auto"):
            raise ValueError("output_mode must be delta, cumulative, or auto")
        if choice_index < 0:
            raise ValueError("choice_index must be non-negative")
        self.engine = engine
        self.sampling_params_factory = sampling_params_factory
        self.prompt_renderer = prompt_renderer
        self.output_mode = output_mode
        self.choice_index = choice_index
        self.prompt_token_counter = prompt_token_counter
        self.name = name
        if max_context_tokens is None:
            max_context_tokens = getattr(getattr(engine, "model_config", None), "max_model_len", None)
        self.capabilities = EngineCapabilities(
            prompt=True,
            chat=prompt_renderer is not None,
            token_ids=True,
            logprobs=True,
            prefix_cache=prefix_cache,
            cache_read_reporting=True,
            max_context_tokens=max_context_tokens,
        )

    async def prompt_token_count(self, request: InferenceRequest) -> int:
        if self.prompt_token_counter is not None:
            return int(await _resolve(self.prompt_token_counter(request)))
        prompt = await self._prompt(request)
        if isinstance(prompt, Mapping) and "prompt_token_ids" in prompt:
            return len(prompt["prompt_token_ids"])
        raise ValueError(
            "vLLM prompt counting requires prompt_token_counter or rendered prompt_token_ids"
        )

    def _default_sampling_params(self, request: InferenceRequest) -> tuple[Any, OutputMode]:
        try:
            from vllm import SamplingParams
        except ImportError as error:  # pragma: no cover - requires optional vLLM
            raise ImportError(
                "native vLLM use requires vLLM or a sampling_params_factory"
            ) from error

        nested = request.extra.get("sampling_params")
        if nested is not None and not isinstance(nested, Mapping):
            raise TypeError("extra['sampling_params'] must be a mapping")
        options = dict(nested or {})
        if request.max_tokens is not None:
            options.setdefault("max_tokens", request.max_tokens)
        if request.temperature is not None:
            options.setdefault("temperature", request.temperature)
        if request.stop:
            options.setdefault("stop", list(request.stop))

        mode: OutputMode = "cumulative"
        try:
            from vllm.sampling_params import RequestOutputKind
        except ImportError:
            pass
        else:
            options.setdefault("output_kind", RequestOutputKind.DELTA)
            mode = "delta"
        return SamplingParams(**options), mode

    async def _sampling_params(self, request: InferenceRequest) -> tuple[Any, OutputMode]:
        if self.sampling_params_factory is None:
            params, default_mode = self._default_sampling_params(request)
            return params, self.output_mode if self.output_mode != "auto" else default_mode
        params = await _resolve(self.sampling_params_factory(request))
        return params, self.output_mode

    async def _prompt(self, request: InferenceRequest) -> Any:
        if request.prompt is not None:
            return request.prompt
        if self.prompt_renderer is None:
            raise ValueError("chat requests require prompt_renderer")
        return await _resolve(self.prompt_renderer(request))

    async def _abort(self, request_id: str) -> None:
        abort = getattr(self.engine, "abort", None)
        if abort is not None:
            await _resolve(abort(request_id))

    async def stream(self, request: InferenceRequest) -> AsyncIterator[StreamChunk]:
        prompt = await self._prompt(request)
        sampling_params, output_mode = await self._sampling_params(request)
        generate_kwargs = request.extra.get("generate_kwargs") or {}
        if not isinstance(generate_kwargs, Mapping):
            raise TypeError("extra['generate_kwargs'] must be a mapping")

        results = await _resolve(
            self.engine.generate(
                prompt=prompt,
                sampling_params=sampling_params,
                request_id=request.request_id,
                **dict(generate_kwargs),
            )
        )
        previous_text = ""
        previous_ids: tuple[int, ...] = ()
        completed = False
        exhausted = False
        try:
            async for request_output in results:
                outputs = getattr(request_output, "outputs", ())
                if self.choice_index >= len(outputs):
                    continue
                output = outputs[self.choice_index]
                text = str(getattr(output, "text", ""))
                token_ids = tuple(int(item) for item in getattr(output, "token_ids", ()))

                cumulative = output_mode == "cumulative"
                if output_mode == "auto":
                    cumulative = (
                        bool(previous_text)
                        and text.startswith(previous_text)
                        or bool(previous_ids)
                        and token_ids[: len(previous_ids)] == previous_ids
                    )
                if cumulative:
                    delta_text = (
                        text[len(previous_text) :]
                        if text.startswith(previous_text)
                        else text
                    )
                    delta_ids = (
                        token_ids[len(previous_ids) :]
                        if token_ids[: len(previous_ids)] == previous_ids
                        else token_ids
                    )
                    previous_text = text
                    previous_ids = token_ids
                else:
                    delta_text = text
                    delta_ids = token_ids
                    previous_text += text
                    previous_ids += token_ids

                finished = bool(getattr(request_output, "finished", False))
                finish_reason = getattr(output, "finish_reason", None)
                if finished and finish_reason is None:
                    finish_reason = "stop"
                completed = completed or finished
                prompt_ids = getattr(request_output, "prompt_token_ids", ()) or ()
                cached_tokens = getattr(request_output, "num_cached_tokens", None)
                yield StreamChunk(
                    text=delta_text,
                    token_ids=delta_ids,
                    logprobs=_native_logprobs(getattr(output, "logprobs", None)),
                    finish_reason=finish_reason,
                    usage={
                        "prompt_tokens": len(prompt_ids),
                        "completion_tokens": len(previous_ids),
                        "total_tokens": len(prompt_ids) + len(previous_ids),
                        **(
                            {"cache_read_tokens": int(cached_tokens)}
                            if cached_tokens is not None
                            else {}
                        ),
                    },
                    raw=request_output,
                )
            exhausted = True
        finally:
            if not completed and not exhausted:
                await self._abort(request.request_id)
            close = getattr(results, "aclose", None)
            if close is not None:
                await close()
