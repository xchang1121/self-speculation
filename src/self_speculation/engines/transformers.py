"""In-process Hugging Face Transformers streaming adapter."""

from __future__ import annotations

import asyncio
import inspect
import threading
import types
import weakref
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from typing import Any

from ..drafts import (
    BoundaryDraftStore,
    DraftReceipt,
    DraftRequest,
    DraftVerificationOutcome,
)
from ..models import EngineCapabilities, InferenceRequest, StreamChunk


TransformersPromptRenderer = Callable[
    [InferenceRequest],
    str | Awaitable[str],
]
TransformersDraftRequestPredicate = Callable[[InferenceRequest], bool]


_MODEL_LOCKS: weakref.WeakKeyDictionary[Any, threading.RLock] = (
    weakref.WeakKeyDictionary()
)
_MODEL_LOCKS_GUARD = threading.Lock()
_FALLBACK_MODEL_LOCK = threading.RLock()


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


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ImportError(
            "Transformers draft injection requires: "
            "pip install 'self-speculation[transformers]'"
        ) from error
    return torch


def _model_lock(model: Any) -> threading.RLock:
    try:
        with _MODEL_LOCKS_GUARD:
            lock = _MODEL_LOCKS.get(model)
            if lock is None:
                lock = threading.RLock()
                _MODEL_LOCKS[model] = lock
            return lock
    except TypeError:
        return _FALLBACK_MODEL_LOCK


class TransformersBoundaryCandidateGenerator:
    """Offer request-scoped boundary drafts to Transformers assisted decoding."""

    requires_model_outputs = False

    def __init__(
        self,
        store: BoundaryDraftStore,
        request_id: str,
        *,
        max_tokens: int,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        self.store = store
        self.request_id = request_id
        self.max_tokens = max_tokens
        self.proposal_count = 0
        self.proposed_tokens = 0
        self.accepted_tokens = 0
        self._last_candidate_length = 0

    def get_candidates(self, input_ids: Any, **kwargs: Any) -> tuple[Any, None]:
        del kwargs
        if len(input_ids.shape) != 2 or int(input_ids.shape[0]) != 1:
            raise ValueError(
                "Transformers boundary drafts require batch size 1 token IDs"
            )
        sequence_length = int(input_ids.shape[1])
        proposal = self.store.offer(
            self.request_id,
            input_ids[0].tolist(),
            sequence_length=sequence_length,
            max_tokens=self.max_tokens,
        )
        if proposal is None:
            self._last_candidate_length = 0
            return input_ids, None

        torch = _torch()
        candidate = torch.tensor(
            [proposal.token_ids],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        self._last_candidate_length = len(proposal.token_ids)
        self.proposal_count += 1
        self.proposed_tokens += self._last_candidate_length
        return torch.cat((input_ids, candidate), dim=-1), None

    def update_candidate_strategy(
        self,
        input_ids: Any,
        scores: Any,
        num_matches: Any,
    ) -> None:
        del input_ids, scores
        accepted = min(int(num_matches), self._last_candidate_length)
        self.accepted_tokens += max(0, accepted)
        if self._last_candidate_length:
            self.store.observe_acceptance(self.request_id, max(0, accepted))
        self._last_candidate_length = 0


def _boundary_assisted_generate(
    model: Any,
    input_ids: Any,
    logits_processor: Any,
    stopping_criteria: Any,
    generation_config: Any,
    draft_candidate_generator: TransformersBoundaryCandidateGenerator,
    self_speculation_streamer: Any = None,
    synced_gpus: bool = False,
    inputs_tensor: Any = None,
    **model_kwargs: Any,
) -> Any:
    """Run Transformers' target verifier with an external candidate source."""

    if not bool(getattr(generation_config, "use_cache", False)):
        raise ValueError("Transformers draft injection requires use_cache=True")
    if int(getattr(generation_config, "num_beams", 1)) != 1:
        raise ValueError("Transformers draft injection does not support beam search")
    if int(getattr(generation_config, "num_return_sequences", 1)) != 1:
        raise ValueError(
            "Transformers draft injection requires num_return_sequences=1"
        )

    lock = _model_lock(model)
    with lock:
        instance_attributes = getattr(model, "__dict__", {})
        had_instance_getter = "_get_candidate_generator" in instance_attributes
        previous_instance_getter = instance_attributes.get("_get_candidate_generator")

        def get_candidate_generator(current_model: Any, **kwargs: Any) -> Any:
            del current_model, kwargs
            return draft_candidate_generator

        model._get_candidate_generator = types.MethodType(
            get_candidate_generator,
            model,
        )
        try:
            return model._assisted_decoding(
                input_ids,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                streamer=self_speculation_streamer,
                inputs_tensor=inputs_tensor,
                **model_kwargs,
            )
        finally:
            if had_instance_getter:
                model._get_candidate_generator = previous_instance_getter
            else:
                delattr(model, "_get_candidate_generator")


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
            self._end_lock = threading.Lock()
            self._ended = False

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

        def end(self) -> None:
            with self._end_lock:
                if self._ended:
                    return
                self._ended = True
                super().end()

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
        draft_store: BoundaryDraftStore | None = None,
        max_draft_tokens: int = 28,
        draft_request_predicate: TransformersDraftRequestPredicate | None = None,
        name: str = "transformers",
    ) -> None:
        if model is None:
            raise ValueError("model is required")
        if tokenizer is None:
            raise ValueError("tokenizer is required")
        if max_draft_tokens <= 0:
            raise ValueError("max_draft_tokens must be positive")
        self.model = model
        self.tokenizer = tokenizer
        self.prompt_renderer = prompt_renderer
        self.generation_kwargs = dict(generation_kwargs or {})
        self.add_special_tokens = add_special_tokens
        self.skip_special_tokens = skip_special_tokens
        self.clean_up_tokenization_spaces = clean_up_tokenization_spaces
        self.draft_store = draft_store
        self.max_draft_tokens = max_draft_tokens
        self.draft_request_predicate = draft_request_predicate or (
            lambda request: not request.request_id.endswith(":fork")
        )
        self.name = name
        self.capabilities = EngineCapabilities(
            prompt=True,
            chat=True,
            token_ids=True,
            prefix_cache=False,
            draft_feedback=draft_store is not None,
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

    def tokenize_continuation(self, text: str) -> tuple[int, ...]:
        encode = getattr(self.tokenizer, "encode", None)
        if encode is None:
            encoded = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        else:
            encoded = encode(text, add_special_tokens=False)
        if hasattr(encoded, "tolist"):
            encoded = encoded.tolist()
        if encoded and isinstance(encoded[0], list):
            if len(encoded) != 1:
                raise ValueError("TransformersEngine only supports batch size 1")
            encoded = encoded[0]
        return tuple(int(token_id) for token_id in encoded)

    async def prompt_token_count(self, request: InferenceRequest) -> int:
        return len(self.tokenize_text(await self.render_prompt(request)))

    async def submit(self, draft: DraftRequest) -> DraftReceipt:
        if self.draft_store is None:
            raise RuntimeError("TransformersEngine has no configured draft_store")
        return self.draft_store.register(draft)

    async def clear(
        self, request_id: str
    ) -> DraftVerificationOutcome | None:
        if self.draft_store is None:
            return None
        return self.draft_store.take_outcome(request_id)

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
        if self.draft_store is not None and self.draft_request_predicate(request):
            if "custom_generate" in options:
                raise ValueError(
                    "generate_kwargs['custom_generate'] conflicts with draft injection"
                )
            if options.get("use_cache") is False:
                raise ValueError(
                    "Transformers draft injection requires use_cache=True"
                )
            options.setdefault("use_cache", True)
            options["custom_generate"] = _boundary_assisted_generate
            options["draft_candidate_generator"] = (
                TransformersBoundaryCandidateGenerator(
                    self.draft_store,
                    request.request_id,
                    max_tokens=self.max_draft_tokens,
                )
            )
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
        if "draft_candidate_generator" in options:
            # Transformers filters standard generation-mode arguments when a
            # custom callable is used, so pass the same streamer under a
            # callable-specific name as well.
            options["self_speculation_streamer"] = streamer

        def generate() -> Any:
            try:
                return self.model.generate(**dict(inputs), **options)
            except BaseException:
                streamer.end()
                raise

        generation_task = asyncio.create_task(asyncio.to_thread(generate))
        iterator = iter(streamer)
        iteration_end = object()
        next_task: asyncio.Task[Any] | None = asyncio.create_task(
            asyncio.to_thread(next, iterator, iteration_end)
        )
        completed = False
        try:
            while True:
                assert next_task is not None
                done, _ = await asyncio.wait(
                    (next_task, generation_task),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if generation_task in done:
                    # All standard generation paths should end the streamer.
                    # Ending is idempotent here and also unblocks custom paths
                    # that return or fail before doing so themselves.
                    streamer.end()
                if next_task not in done:
                    continue
                chunk = next_task.result()
                if chunk is iteration_end:
                    next_task = None
                    break
                if not isinstance(chunk, StreamChunk):
                    raise TypeError("Transformers streamer produced an invalid chunk")
                yield chunk
                next_task = asyncio.create_task(
                    asyncio.to_thread(next, iterator, iteration_end)
                )
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
            streamer.end()
            if next_task is not None:
                try:
                    await next_task
                except BaseException:
                    pass
            if not completed:
                try:
                    await generation_task
                except BaseException:
                    pass
