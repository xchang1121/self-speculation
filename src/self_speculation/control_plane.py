"""Structured control-plane primitives shared by agent and engine adapters."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from .decoding import default_decoder
from .drafts import (
    DraftBoundary,
    DraftBundle,
    DraftBundleFeedback,
    DraftReceipt,
    DraftRequest,
    DraftTokenizer,
    DraftVerificationOutcome,
    ToolCallDraftBuilder,
    default_draft_boundary,
    format_tool_call_draft,
)
from .engines import InferenceEngine, validate_request
from .forks import PrefixForkBuilder, PromptRenderer
from .models import InferenceRequest, StreamSnapshot, ToolCall


class ControlRequestClosedError(RuntimeError):
    """A late control update targeted a request that was already cleared."""


def _positive_integer(value: Any, *, field: str, fallback: int) -> int:
    if value is None:
        return fallback
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{field} must be an integer") from error
    if result <= 0:
        raise ValueError(f"{field} must be positive")
    return result


def _array(value: Any, *, field: str) -> tuple[Any, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array")
    return tuple(value)


@dataclass(slots=True)
class CandidateBundleBuilder:
    """Turn ranked concrete tool calls into target-tokenizer draft choices."""

    tokenizer: DraftTokenizer
    max_candidates: int = 64
    max_draft_tokens: int = 28
    default_format: str = "tagged_json"

    def __post_init__(self) -> None:
        if self.max_candidates <= 0:
            raise ValueError("max_candidates must be positive")
        if self.max_draft_tokens <= 0:
            raise ValueError("max_draft_tokens must be positive")
        if not self.default_format.strip():
            raise ValueError("default_format must not be empty")

    async def build(self, payload: Mapping[str, Any]) -> DraftBundle:
        request_id = str(payload.get("request_id") or "").strip()
        if not request_id:
            raise ValueError("request_id must not be empty")
        raw_candidates = payload.get("candidates")
        if not isinstance(raw_candidates, list) or not raw_candidates:
            raise ValueError("candidates must be a non-empty array")
        if len(raw_candidates) > self.max_candidates:
            raise ValueError(
                f"candidates exceeds the server limit of {self.max_candidates}"
            )

        max_draft_tokens = min(
            _positive_integer(
                payload.get("max_draft_tokens"),
                field="max_draft_tokens",
                fallback=self.max_draft_tokens,
            ),
            self.max_draft_tokens,
        )
        format_name = str(
            payload.get("format") or self.default_format
        ).strip()
        if not format_name:
            raise ValueError("format must not be empty")
        boundary_text_value = payload.get("boundary")
        boundary_text = (
            str(boundary_text_value)
            if boundary_text_value is not None
            else None
        )
        if boundary_text == "":
            raise ValueError("boundary must not be empty")
        prompt_token_count = (
            int(payload["prompt_token_count"])
            if payload.get("prompt_token_count") is not None
            else None
        )

        def boundary(calls: tuple[ToolCall, ...]) -> DraftBoundary:
            if boundary_text is not None:
                return DraftBoundary(text=boundary_text)
            return default_draft_boundary(calls)

        builder = ToolCallDraftBuilder(
            formatter=format_tool_call_draft,
            tokenizer=self.tokenizer,
            boundary_resolver=boundary,
            max_draft_tokens=max_draft_tokens,
        )
        drafts = []
        for index, raw_candidate in enumerate(raw_candidates):
            if not isinstance(raw_candidate, Mapping):
                raise ValueError("every candidate must be an object")
            raw_call = raw_candidate.get("tool_call")
            if not isinstance(raw_call, Mapping):
                raise ValueError("candidate.tool_call must be an object")
            name = str(raw_call.get("name") or "").strip()
            if not name:
                raise ValueError("candidate.tool_call.name must not be empty")
            candidate_id = str(raw_candidate.get("id") or index)
            score = raw_candidate.get("score") or {}
            if not isinstance(score, Mapping):
                raise ValueError("candidate.score must be an object")
            call = ToolCall(
                name=name,
                arguments=raw_call.get("arguments", {}),
                index=0,
                format=format_name,
            )
            draft = await builder.build_for_request(
                (call,),
                request_id=request_id,
                prompt_token_count=prompt_token_count,
                metadata={
                    "candidate_id": candidate_id,
                    "candidate_index": index,
                    "sources": _array(
                        raw_candidate.get("sources"), field="candidate.sources"
                    ),
                    "provenance": _array(
                        raw_candidate.get("provenance"),
                        field="candidate.provenance",
                    ),
                    "score": dict(score),
                },
            )
            drafts.append(draft)
        return DraftBundle(
            request_id=request_id,
            drafts=tuple(drafts),
            metadata={
                "version": payload.get("version", 1),
                "model": dict(payload.get("model") or {}),
            },
        )


@dataclass(slots=True)
class SnapshotForkRunner:
    """Run one D1 fork from an externally captured Actor stream snapshot."""

    engine: InferenceEngine
    tokenizer: DraftTokenizer
    prompt_renderer: PromptRenderer | None = None
    default_format: str = "tagged_json"
    default_boundary: str = "<tool_call>"
    max_draft_tokens: int = 28

    def __post_init__(self) -> None:
        if self.max_draft_tokens <= 0:
            raise ValueError("max_draft_tokens must be positive")

    async def run(self, payload: Mapping[str, Any]) -> DraftRequest:
        fork_started_at = time.perf_counter()
        request_id = str(payload.get("request_id") or "").strip()
        if not request_id:
            raise ValueError("request_id must not be empty")
        context = payload.get("context")
        snapshot_payload = payload.get("snapshot")
        options = payload.get("options") or {}
        if not isinstance(context, Mapping):
            raise ValueError("context must be an object")
        if not isinstance(snapshot_payload, Mapping):
            raise ValueError("snapshot must be an object")
        if not isinstance(options, Mapping):
            raise ValueError("options must be an object")

        main_request = _inference_request(request_id, payload, context)
        snapshot = StreamSnapshot(
            generated_text=str(snapshot_payload.get("generated_text") or ""),
            content=str(snapshot_payload.get("content") or ""),
            reasoning=str(snapshot_payload.get("reasoning") or ""),
            chunk_count=int(snapshot_payload.get("chunk_count") or 0),
            output_chunk_count=int(
                snapshot_payload.get("output_chunk_count") or 0
            ),
            token_count=int(snapshot_payload.get("token_count") or 0),
        )
        forced_prefix = str(
            options.get("forced_prefix") or self.default_boundary
        )
        max_tokens = _positive_integer(
            options.get("max_tokens"), field="max_tokens", fallback=128
        )
        temperature = float(options.get("temperature", 0.0))
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        require_logprobs = bool(options.get("require_logprobs", False))
        if require_logprobs and not self.engine.capabilities.logprobs:
            raise ValueError(
                f"engine {self.engine.name!r} does not expose token logprobs"
            )
        builder = PrefixForkBuilder(
            forced_prefix=forced_prefix,
            prompt_renderer=self.prompt_renderer,
            max_tokens=max_tokens,
            temperature=temperature,
            extra=(
                {"logprobs": True, "top_logprobs": 1}
                if require_logprobs
                else {}
            ),
        )
        fork_request = await builder.build(main_request, snapshot)
        validate_request(self.engine, fork_request)
        fork_built_at = time.perf_counter()
        decoder_name = str(options.get("decoder") or "auto")
        decoder = default_decoder(decoder_name, initial_text=forced_prefix)
        iterator = self.engine.stream(fork_request).__aiter__()
        decoded: list[ToolCall] = []
        first_chunk_ms: float | None = None
        output_chunks = 0
        decoded_tokens = 0
        logprob_values: list[float] = []
        try:
            async for chunk in iterator:
                if first_chunk_ms is None:
                    first_chunk_ms = (time.perf_counter() - fork_built_at) * 1000
                output_chunks += int(chunk.has_output)
                decoded_tokens += max(len(chunk.token_ids), len(chunk.logprobs))
                logprob_values.extend(
                    value.logprob
                    for value in chunk.logprobs
                    if value.logprob is not None and math.isfinite(value.logprob)
                )
                decoded.extend(decoder.feed(chunk))
                if decoded:
                    break
            if not decoded:
                decoded.extend(decoder.finish())
        finally:
            close = getattr(iterator, "aclose", None)
            if close is not None:
                await close()
        if not decoded:
            raise ValueError("self-speculation fork produced no complete tool call")

        format_name = str(
            options.get("draft_format") or self.default_format
        ).strip()
        boundary_text = str(
            options.get("draft_boundary") or self.default_boundary
        )
        calls = tuple(
            ToolCall(
                name=call.name,
                arguments=call.arguments,
                call_id=call.call_id,
                index=call.index,
                format=format_name,
                raw=call.raw,
            )
            for call in decoded
        )
        draft_builder = ToolCallDraftBuilder(
            formatter=format_tool_call_draft,
            tokenizer=self.tokenizer,
            boundary_resolver=lambda _: DraftBoundary(text=boundary_text),
            max_draft_tokens=min(
                _positive_integer(
                    options.get("max_draft_tokens"),
                    field="max_draft_tokens",
                    fallback=self.max_draft_tokens,
                ),
                self.max_draft_tokens,
            ),
        )
        candidate_id = "self:" + hashlib.sha256(
            json.dumps(
                [(call.name, call.arguments) for call in calls],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=repr,
            ).encode("utf-8")
        ).hexdigest()[:16]
        fork_completed_at = time.perf_counter()
        logprob_observation = (
            {
                "token_count": len(logprob_values),
                "mean": sum(logprob_values) / len(logprob_values),
                "minimum": min(logprob_values),
            }
            if logprob_values
            else {"token_count": 0}
        )
        return await draft_builder.build_for_request(
            calls,
            request_id=request_id,
            metadata={
                "candidate_id": candidate_id,
                "candidate_index": "self",
                "sources": ("self-speculation",),
                "fork_request_id": fork_request.request_id,
                "fork": {
                    "build_ms": (fork_built_at - fork_started_at) * 1000,
                    "decode_ms": (fork_completed_at - fork_built_at) * 1000,
                    "total_ms": (fork_completed_at - fork_started_at) * 1000,
                    "first_chunk_ms": first_chunk_ms,
                    "output_chunks": output_chunks,
                    "decoded_tokens": decoded_tokens,
                    "logprobs": logprob_observation,
                },
            },
        )


@dataclass(slots=True)
class _ControlState:
    external: DraftBundle | None = None
    self_draft: DraftRequest | None = None
    closed: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class SelfSpeculationControlPlane:
    """Merge external actions and a D1 self-fork into one D3 draft bundle."""

    def __init__(
        self,
        feedback: DraftBundleFeedback,
        candidate_builder: CandidateBundleBuilder,
        *,
        fork_runner: SnapshotForkRunner | None = None,
        max_closed_requests: int = 4096,
    ) -> None:
        if max_closed_requests <= 0:
            raise ValueError("max_closed_requests must be positive")
        self.feedback = feedback
        self.candidate_builder = candidate_builder
        self.fork_runner = fork_runner
        self._states: dict[str, _ControlState] = {}
        self._states_lock = asyncio.Lock()
        self._closed_requests: dict[str, None] = {}
        self._max_closed_requests = max_closed_requests

    async def submit_candidates(
        self, payload: Mapping[str, Any]
    ) -> DraftReceipt:
        request_id = _payload_request_id(payload)
        state = await self._state_for(request_id)
        bundle = await self.candidate_builder.build(payload)
        async with state.lock:
            self._ensure_open(request_id, state)
            state.external = bundle
            combined = self._combined(bundle.request_id, state)
            receipt = await self.feedback.submit_bundle(combined)
            return _receipt_with_bundle_observation(receipt, combined)

    async def fork(self, payload: Mapping[str, Any]) -> DraftReceipt:
        if self.fork_runner is None:
            raise RuntimeError("sidecar fork runner is not configured")
        request_id = _payload_request_id(payload)
        state = await self._state_for(request_id)
        draft = await self.fork_runner.run(payload)
        async with state.lock:
            self._ensure_open(request_id, state)
            state.self_draft = draft
            combined = self._combined(draft.request_id, state)
            receipt = await self.feedback.submit_bundle(combined)
            return _receipt_with_bundle_observation(receipt, combined)

    async def clear(
        self, request_id: str
    ) -> DraftVerificationOutcome | None:
        request_id = request_id.strip()
        if not request_id:
            raise ValueError("request_id must not be empty")
        async with self._states_lock:
            state = self._states.pop(request_id, None)
            if state is not None:
                state.closed = True
            self._closed_requests.pop(request_id, None)
            self._closed_requests[request_id] = None
            while len(self._closed_requests) > self._max_closed_requests:
                del self._closed_requests[next(iter(self._closed_requests))]
        if state is not None:
            async with state.lock:
                return await self.feedback.clear(request_id)
        return await self.feedback.clear(request_id)

    async def _state_for(self, request_id: str) -> _ControlState:
        async with self._states_lock:
            if request_id in self._closed_requests:
                raise ControlRequestClosedError(
                    f"self-speculation request {request_id!r} is already cleared"
                )
            return self._states.setdefault(request_id, _ControlState())

    def _ensure_open(self, request_id: str, state: _ControlState) -> None:
        if state.closed or self._states.get(request_id) is not state:
            raise ControlRequestClosedError(
                f"self-speculation request {request_id!r} is already cleared"
            )

    @staticmethod
    def _combined(request_id: str, state: _ControlState) -> DraftBundle:
        drafts = list(state.external.drafts if state.external is not None else ())
        if state.self_draft is not None:
            drafts.append(state.self_draft)
        distinct: list[DraftRequest] = []
        positions: dict[tuple[object, ...], int] = {}
        for draft in drafts:
            identity = _draft_identity(draft)
            position = positions.get(identity)
            if position is None:
                positions[identity] = len(distinct)
                distinct.append(draft)
            else:
                distinct[position] = _merge_draft_metadata(
                    distinct[position], draft
                )
        return DraftBundle(
            request_id=request_id,
            drafts=tuple(distinct),
            metadata={"sources": "unified"},
        )


def install_self_speculation_routes(
    app: Any,
    control_plane: SelfSpeculationControlPlane,
    *,
    prefix: str = "/self-speculation",
) -> bool:
    """Install the portable agent-facing candidate/fork/clear protocol."""

    if not prefix.startswith("/") or prefix.endswith("/"):
        raise ValueError("route prefix must start with '/' and not end with '/'")
    state = getattr(app, "state", None)
    add_api_route = getattr(app, "add_api_route", None)
    if state is None or not callable(add_api_route):
        raise TypeError("app must be a FastAPI-compatible application")
    marker = "_self_speculation_control_plane_routes"
    if getattr(state, marker, False):
        return False
    try:
        from fastapi import HTTPException
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ImportError(
            "control-plane routes require: pip install 'self-speculation[server]'"
        ) from error

    def receipt_payload(receipt: DraftReceipt) -> dict[str, Any]:
        return {
            "request_id": receipt.request_id,
            "registered": receipt.registered,
            "draft_token_count": receipt.draft_token_count,
            "accepted_token_count": receipt.accepted_token_count,
            "details": dict(receipt.details),
        }

    async def candidates(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return receipt_payload(await control_plane.submit_candidates(payload))
        except (TypeError, ValueError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except ControlRequestClosedError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except Exception as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    async def fork(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return receipt_payload(await control_plane.fork(payload))
        except (TypeError, ValueError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except ControlRequestClosedError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except Exception as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    async def clear(payload: dict[str, Any]) -> dict[str, Any]:
        request_id = str(payload.get("request_id") or "")
        try:
            verification = await control_plane.clear(request_id)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except Exception as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        return {
            "status": "cleared",
            "request_id": request_id,
            **(
                {"verification": verification.to_mapping()}
                if verification is not None
                else {}
            ),
        }

    add_api_route(prefix + "/candidates", candidates, methods=["POST"])
    add_api_route(prefix + "/fork", fork, methods=["POST"])
    add_api_route(prefix + "/clear", clear, methods=["POST"])
    setattr(state, marker, True)
    return True


def _inference_request(
    request_id: str,
    payload: Mapping[str, Any],
    context: Mapping[str, Any],
) -> InferenceRequest:
    model_payload = payload.get("model") or {}
    if not isinstance(model_payload, Mapping):
        raise ValueError("model must be an object")
    model = str(model_payload.get("id") or "").strip() or None
    provider_payload = context.get("provider_payload") or {}
    if not isinstance(provider_payload, Mapping):
        raise ValueError("context.provider_payload must be an object")
    provider_model = str(provider_payload.get("model") or "").strip()
    if provider_model:
        model = provider_model
    prompt = context.get("prompt")
    if prompt is None:
        prompt = provider_payload.get("prompt")
    if prompt is not None:
        return InferenceRequest(
            prompt=str(prompt),
            model=model,
            request_id=request_id,
        )
    raw_messages = provider_payload.get("messages", context.get("messages")) or []
    if not isinstance(raw_messages, list):
        raise ValueError("context.messages must be an array")
    messages = list(raw_messages)
    system_prompt = str(context.get("system_prompt") or "")
    if system_prompt and not provider_payload.get("messages"):
        messages.insert(0, {"role": "system", "content": system_prompt})
    raw_tools = provider_payload.get("tools", context.get("tools")) or []
    if not isinstance(raw_tools, list):
        raise ValueError("context.tools must be an array")
    return InferenceRequest(
        messages=tuple(messages),
        model=model,
        tools=tuple(raw_tools),
        request_id=request_id,
    )


def _payload_request_id(payload: Mapping[str, Any]) -> str:
    request_id = str(payload.get("request_id") or "").strip()
    if not request_id:
        raise ValueError("request_id must not be empty")
    return request_id


def _draft_identity(draft: DraftRequest) -> tuple[object, ...]:
    boundary = draft.boundary
    return (
        ("text", draft.text)
        if draft.text
        else ("tokens", draft.token_ids),
        (
            ("text", boundary.text)
            if boundary is not None and boundary.text
            else (
                "tokens",
                boundary.token_ids if boundary is not None else (),
            )
        ),
    )


def _metadata_array(value: Any) -> tuple[Any, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return ()


def _unique_metadata(values: tuple[Any, ...]) -> tuple[Any, ...]:
    unique: list[Any] = []
    seen: set[str] = set()
    for value in values:
        key = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=repr,
        )
        if key not in seen:
            seen.add(key)
            unique.append(value)
    return tuple(unique)


def _merge_draft_metadata(
    primary: DraftRequest, duplicate: DraftRequest
) -> DraftRequest:
    sources = tuple(
        sorted(
            {
                str(source)
                for source in (
                    *_metadata_array(primary.metadata.get("sources")),
                    *_metadata_array(duplicate.metadata.get("sources")),
                )
                if str(source)
            }
        )
    )
    provenance = _unique_metadata(
        _metadata_array(primary.metadata.get("provenance"))
        + _metadata_array(duplicate.metadata.get("provenance"))
    )
    candidate_ids = _unique_metadata(
        tuple(
            candidate_id
            for candidate_id in (
                primary.metadata.get("candidate_id"),
                duplicate.metadata.get("candidate_id"),
            )
            if candidate_id is not None
        )
        + _metadata_array(primary.metadata.get("candidate_ids"))
        + _metadata_array(duplicate.metadata.get("candidate_ids"))
    )
    metadata = {
        **dict(primary.metadata),
        "sources": sources,
        "source_count": len(sources),
        "candidate_ids": candidate_ids,
    }
    if provenance:
        metadata["provenance"] = provenance
    fork = duplicate.metadata.get("fork") or primary.metadata.get("fork")
    if isinstance(fork, Mapping):
        metadata["fork"] = dict(fork)
    return replace(primary, metadata=metadata)


def _receipt_with_bundle_observation(
    receipt: DraftReceipt, bundle: DraftBundle
) -> DraftReceipt:
    """Attach bounded, JSON-safe candidate diagnostics without changing feedback."""

    candidates = tuple(_draft_observation(draft) for draft in bundle.drafts)
    return replace(
        receipt,
        details={
            **dict(receipt.details),
            "bundle": {
                "candidate_count": len(candidates),
                "candidates": candidates,
            },
        },
    )


def _draft_observation(draft: DraftRequest) -> dict[str, Any]:
    metadata = draft.metadata
    candidate_id = metadata.get("candidate_id")
    candidate_ids = _unique_metadata(
        tuple(value for value in (candidate_id,) if value is not None)
        + _metadata_array(metadata.get("candidate_ids"))
    )
    observation: dict[str, Any] = {
        "candidate_ids": candidate_ids,
        "sources": _metadata_array(metadata.get("sources")),
        "draft_token_count": len(draft.token_ids),
        "tool_calls": tuple(
            {
                "name": call.name,
                "arguments": call.arguments,
                "index": call.index,
                "format": call.format,
            }
            for call in draft.tool_calls
        ),
    }
    score = metadata.get("score")
    if isinstance(score, Mapping):
        observation["score"] = dict(score)
    fork = metadata.get("fork")
    if isinstance(fork, Mapping):
        observation["fork"] = dict(fork)
    return observation


__all__ = [
    "CandidateBundleBuilder",
    "ControlRequestClosedError",
    "SelfSpeculationControlPlane",
    "SnapshotForkRunner",
    "install_self_speculation_routes",
]
