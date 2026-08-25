"""vLLM custom proposer integration with stable request-ID routing.

The module deliberately imports vLLM only when installation is requested, so
the base package remains usable with other engines and on non-CUDA hosts.
"""

from __future__ import annotations

import asyncio
import inspect
import os
import threading
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import asdict
from functools import wraps
from typing import Any

from ..control_plane import CandidateBundleBuilder
from ..drafts import (
    BoundaryDraftStore,
    DraftBundle,
    DraftBoundary,
    DraftReceipt,
    DraftRequest,
    DraftVerificationOutcome,
    HTTPDraftFeedback,
    normalize_draft_receipt,
)


_HOOK_MARKER = "_self_speculation_request_id_hook"
_RPC_MARKER = "_self_speculation_worker_rpc"
REGISTER_DRAFT_RPC = "self_speculation_register_draft"
REGISTER_DRAFT_BUNDLE_RPC = "self_speculation_register_draft_bundle"
CLEAR_DRAFT_RPC = "self_speculation_clear_draft"
DRAFT_STATUS_RPC = "self_speculation_draft_status"
EngineClientResolver = Callable[[Any], Any | Awaitable[Any]]
_PLUGIN_ENGINE_CLIENT_STATE = "_self_speculation_engine_client"


class VLLMIntegrationError(RuntimeError):
    pass


class VLLMRequestNotActiveError(VLLMIntegrationError):
    """The control request arrived before vLLM admitted the main request."""


def _worker_proposer(worker: Any) -> Any | None:
    runner = getattr(worker, "model_runner", None)
    proposer = getattr(runner, "drafter", None)
    if proposer is None or not hasattr(proposer, "register_draft"):
        return None
    return proposer


def _worker_prompt_token_count(runner: Any, request_id: str) -> int:
    requests = getattr(runner, "requests", None)
    request_state = requests.get(request_id) if isinstance(requests, Mapping) else None
    if request_state is None:
        raise VLLMIntegrationError(
            f"active vLLM request state not found for {request_id!r}"
        )
    count = getattr(request_state, "num_prompt_tokens", None)
    if count is not None:
        return int(count)
    prompt_token_ids = getattr(request_state, "prompt_token_ids", None)
    if prompt_token_ids is not None:
        return len(prompt_token_ids)
    prompt_embeds = getattr(request_state, "prompt_embeds", None)
    shape = getattr(prompt_embeds, "shape", None)
    if shape:
        return int(shape[0])
    raise VLLMIntegrationError(
        f"prompt length is unavailable for active request {request_id!r}"
    )


def _worker_request_id(runner: Any, external_request_id: str) -> str:
    requests = getattr(runner, "requests", None)
    if not isinstance(requests, Mapping):
        raise VLLMIntegrationError("active vLLM request mapping is unavailable")
    if external_request_id in requests:
        return external_request_id

    bases = (
        external_request_id,
        f"cmpl-{external_request_id}-0",
        f"chatcmpl-{external_request_id}",
    )
    matches = {
        str(internal_request_id)
        for internal_request_id in requests
        if any(
            str(internal_request_id) == base
            or str(internal_request_id).startswith(base + "-")
            for base in bases
        )
    }
    if len(matches) == 1:
        return matches.pop()
    if not matches:
        raise VLLMRequestNotActiveError(
            f"active vLLM request not found for external ID {external_request_id!r}"
        )
    raise VLLMIntegrationError(
        f"external vLLM request ID {external_request_id!r} is ambiguous"
    )


def install_vllm_worker_rpc(worker_class: type[Any] | None = None) -> bool:
    """Install request-scoped draft control methods on a vLLM V1 worker."""

    if worker_class is None:
        try:
            from vllm.v1.worker.gpu_worker import Worker
        except ImportError as error:  # pragma: no cover - requires optional vLLM
            raise ImportError("vLLM worker RPC integration requires vLLM V1") from error
        worker_class = Worker

    if getattr(worker_class, _RPC_MARKER, False):
        return False
    method_names = (
        REGISTER_DRAFT_RPC,
        REGISTER_DRAFT_BUNDLE_RPC,
        CLEAR_DRAFT_RPC,
        DRAFT_STATUS_RPC,
    )
    collisions = [name for name in method_names if hasattr(worker_class, name)]
    if collisions:
        raise RuntimeError(
            "vLLM worker already defines self-speculation RPC methods: "
            + ", ".join(collisions)
        )

    def register(
        worker: Any,
        request_id: str,
        draft_token_ids: list[int],
        boundary_token_ids: list[int],
        prompt_token_count: int | None = None,
    ) -> dict[str, Any]:
        proposer = _worker_proposer(worker)
        if proposer is None:
            return {"status": "skipped", "reason": "no_boundary_proposer"}
        try:
            internal_request_id = _worker_request_id(
                worker.model_runner, request_id
            )
            if prompt_token_count is None:
                prompt_token_count = _worker_prompt_token_count(
                    worker.model_runner, internal_request_id
                )
            return proposer.register_draft(
                internal_request_id,
                draft_token_ids,
                boundary_token_ids,
                prompt_token_count,
                external_request_id=request_id,
            )
        except VLLMRequestNotActiveError:
            return {"status": "pending", "reason": "request_not_active"}
        except (TypeError, ValueError, VLLMIntegrationError) as error:
            return {"status": "error", "error": str(error)}

    def register_bundle(
        worker: Any,
        request_id: str,
        draft_token_ids: list[list[int]],
        boundary_token_ids: list[list[int]],
        prompt_token_count: int | None = None,
        candidate_ids: list[str | None] | None = None,
    ) -> dict[str, Any]:
        proposer = _worker_proposer(worker)
        if proposer is None or not hasattr(proposer, "register_draft_bundle"):
            return {"status": "skipped", "reason": "no_boundary_proposer"}
        try:
            internal_request_id = _worker_request_id(
                worker.model_runner, request_id
            )
            if prompt_token_count is None:
                prompt_token_count = _worker_prompt_token_count(
                    worker.model_runner, internal_request_id
                )
            return proposer.register_draft_bundle(
                internal_request_id,
                draft_token_ids,
                boundary_token_ids,
                prompt_token_count,
                candidate_ids=candidate_ids,
                external_request_id=request_id,
            )
        except VLLMRequestNotActiveError:
            return {"status": "pending", "reason": "request_not_active"}
        except (TypeError, ValueError, VLLMIntegrationError) as error:
            return {"status": "error", "error": str(error)}

    def clear(worker: Any, request_id: str) -> dict[str, Any]:
        proposer = _worker_proposer(worker)
        if proposer is None:
            return {"status": "skipped", "reason": "no_boundary_proposer"}
        return proposer.clear_request(request_id)

    def status(worker: Any) -> dict[str, Any]:
        proposer = _worker_proposer(worker)
        if proposer is None:
            return {"status": "skipped", "reason": "no_boundary_proposer"}
        return {"status": "ok", **proposer.status()}

    setattr(worker_class, REGISTER_DRAFT_RPC, register)
    setattr(worker_class, REGISTER_DRAFT_BUNDLE_RPC, register_bundle)
    setattr(worker_class, CLEAR_DRAFT_RPC, clear)
    setattr(worker_class, DRAFT_STATUS_RPC, status)
    setattr(worker_class, _RPC_MARKER, True)
    return True


async def _invoke_rpc(callback: Any, *args: Any, **kwargs: Any) -> Any:
    if inspect.iscoroutinefunction(callback):
        return await callback(*args, **kwargs)
    result = await asyncio.to_thread(callback, *args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


def _rpc_results(value: Any) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise VLLMIntegrationError("vLLM collective_rpc must return worker results")
    results: list[Mapping[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise VLLMIntegrationError("vLLM worker result must be a mapping")
        status = str(item.get("status", "")).lower()
        if item.get("error") or status in {"error", "failed", "failure"}:
            raise VLLMIntegrationError(str(item.get("error") or item))
        results.append(item)
    if not results:
        raise VLLMIntegrationError("vLLM collective_rpc returned no workers")
    return tuple(results)


async def _engine_token_ids(engine_client: Any, text: str) -> tuple[int, ...]:
    renderer = getattr(engine_client, "renderer", None)
    tokenizer = getattr(renderer, "tokenizer", None)
    if tokenizer is None and renderer is not None:
        getter = getattr(renderer, "get_tokenizer", None)
        if callable(getter):
            tokenizer = getter()
            if inspect.isawaitable(tokenizer):
                tokenizer = await tokenizer
    if tokenizer is None:
        input_processor = getattr(engine_client, "input_processor", None)
        tokenizer = getattr(input_processor, "tokenizer", None)
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        raise VLLMIntegrationError(
            "vLLM engine client does not expose its target tokenizer"
        )
    encoded = encode(text, add_special_tokens=False)
    if inspect.isawaitable(encoded):
        encoded = await encoded
    if isinstance(encoded, Mapping):
        encoded = encoded.get("input_ids")
    elif hasattr(encoded, "ids"):
        encoded = encoded.ids
    if not isinstance(encoded, Sequence) or isinstance(encoded, (str, bytes)):
        raise VLLMIntegrationError("vLLM tokenizer.encode returned invalid token IDs")
    if encoded and isinstance(encoded[0], Sequence):
        raise VLLMIntegrationError("vLLM tokenizer.encode must return one token row")
    return tuple(int(token) for token in encoded)


def _registered_results(
    results: tuple[Mapping[str, Any], ...],
) -> tuple[Mapping[str, Any], ...]:
    registered = tuple(
        result
        for result in results
        if str(result.get("status", "")).lower() in {"ok", "registered"}
        and bool(result.get("registered", True))
    )
    if registered:
        return registered
    if any(result.get("reason") == "request_not_active" for result in results):
        raise VLLMRequestNotActiveError("vLLM main request is not active yet")
    raise VLLMIntegrationError(
        "no vLLM worker has VLLMBoundaryProposer installed"
    )


class VLLMCollectiveRPCDraftFeedback:
    """Feed D3 drafts to an in-process vLLM engine or engine client."""

    name = "vllm-collective-rpc-draft-feedback"

    def __init__(self, engine_client: Any, *, timeout: float | None = 30.0) -> None:
        collective_rpc = getattr(engine_client, "collective_rpc", None)
        if collective_rpc is None or not callable(collective_rpc):
            raise TypeError("engine_client must provide collective_rpc")
        self.engine_client = engine_client
        self.timeout = timeout

    async def _rpc(self, method: str, args: tuple[Any, ...]) -> tuple[Mapping[str, Any], ...]:
        value = await _invoke_rpc(
            self.engine_client.collective_rpc,
            method,
            timeout=self.timeout,
            args=args,
        )
        return _rpc_results(value)

    async def submit(self, draft: DraftRequest) -> DraftReceipt:
        if not draft.token_ids:
            raise ValueError("vLLM draft feedback requires token_ids")
        if draft.boundary is None or not draft.boundary.token_ids:
            raise ValueError("vLLM draft feedback requires boundary token_ids")
        results = await self._rpc(
            REGISTER_DRAFT_RPC,
            (
                draft.request_id,
                list(draft.token_ids),
                list(draft.boundary.token_ids),
                draft.prompt_token_count,
            ),
        )
        registered = _registered_results(results)
        counts = [
            int(result["draft_token_count"])
            for result in registered
            if result.get("draft_token_count") is not None
        ]
        return DraftReceipt(
            request_id=draft.request_id,
            registered=True,
            draft_token_count=min(counts) if counts else len(draft.token_ids),
            details={"worker_results": results},
        )

    async def submit_bundle(self, bundle: DraftBundle) -> DraftReceipt:
        for draft in bundle.drafts:
            if not draft.token_ids:
                raise ValueError("vLLM draft feedback requires token_ids")
            if draft.boundary is None or not draft.boundary.token_ids:
                raise ValueError("vLLM draft feedback requires boundary token_ids")
        prompt_counts = {
            draft.prompt_token_count
            for draft in bundle.drafts
            if draft.prompt_token_count is not None
        }
        if len(prompt_counts) > 1:
            raise ValueError("bundled drafts must use one prompt_token_count")
        candidate_ids = [
            str(draft.metadata["candidate_id"])
            if draft.metadata.get("candidate_id") is not None
            else None
            for draft in bundle.drafts
        ]
        results = await self._rpc(
            REGISTER_DRAFT_BUNDLE_RPC,
            (
                bundle.request_id,
                [list(draft.token_ids) for draft in bundle.drafts],
                [list(draft.boundary.token_ids) for draft in bundle.drafts],
                next(iter(prompt_counts), None),
                candidate_ids,
            ),
        )
        registered = _registered_results(results)
        counts = [
            int(result["draft_token_count"])
            for result in registered
            if result.get("draft_token_count") is not None
        ]
        return DraftReceipt(
            request_id=bundle.request_id,
            registered=True,
            draft_token_count=(
                min(counts)
                if counts
                else max(len(draft.token_ids) for draft in bundle.drafts)
            ),
            details={
                "candidate_count": len(bundle.drafts),
                "worker_results": results,
            },
        )

    async def clear(
        self, request_id: str
    ) -> DraftVerificationOutcome | None:
        results = await self._rpc(CLEAR_DRAFT_RPC, (request_id,))
        outcomes = [
            DraftVerificationOutcome.from_mapping(request_id, verification)
            for result in results
            for verification in (result.get("verification"),)
            if isinstance(verification, Mapping)
        ]
        return max(
            outcomes,
            key=lambda outcome: (
                len(outcome.steps),
                outcome.proposed_tokens,
                -outcome.unresolved_proposals,
            ),
            default=None,
        )

    async def status(self) -> tuple[Mapping[str, Any], ...]:
        return await self._rpc(DRAFT_STATUS_RPC, ())


class VLLMHTTPDraftFeedback(HTTPDraftFeedback):
    """Client for routes installed by ``install_vllm_http_routes``."""

    def __init__(
        self,
        base_url: str,
        *,
        prefix: str = "/self-speculation",
        **kwargs: Any,
    ) -> None:
        if not prefix.startswith("/") or prefix.endswith("/"):
            raise ValueError("route prefix must start with '/' and not end with '/'")
        kwargs.setdefault("submit_path", prefix + "/drafts")
        kwargs.setdefault("clear_path", prefix + "/clear")
        kwargs.setdefault("clear_method", "POST")
        kwargs.setdefault("name", "vllm-http-draft-feedback")
        super().__init__(base_url, **kwargs)
        self.status_path = prefix + "/status"
        self.bundle_submit_path = prefix + "/draft-bundles"

    async def submit_bundle(self, bundle: DraftBundle) -> DraftReceipt:
        response = await self._client.post(
            self._url(self.bundle_submit_path),
            json={
                "request_id": bundle.request_id,
                "drafts": [
                    dict(self.submit_payload(draft))
                    for draft in bundle.drafts
                ],
                "metadata": dict(bundle.metadata),
            },
            headers=self._headers(),
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = self._response_payload(response)
        receipt = normalize_draft_receipt(payload, bundle.drafts[0])
        return DraftReceipt(
            request_id=receipt.request_id,
            registered=receipt.registered,
            draft_token_count=receipt.draft_token_count,
            accepted_token_count=receipt.accepted_token_count,
            details={
                **dict(receipt.details),
                "candidate_count": len(bundle.drafts),
            },
        )

    def clear_payload(self, request_id: str) -> Mapping[str, Any]:
        return {"request_id": request_id}

    async def status(self) -> Mapping[str, Any]:
        response = await self._client.get(
            self.base_url + self.status_path,
            headers=self._headers(),
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = self._response_payload(response)
        if not isinstance(payload, Mapping):
            raise VLLMIntegrationError("vLLM status response must be a mapping")
        return payload


def install_vllm_http_routes(
    app: Any,
    *,
    engine_client_resolver: EngineClientResolver | None = None,
    prefix: str = "/self-speculation",
    timeout: float | None = 30.0,
    registration_wait_timeout: float = 1.0,
    registration_retry_interval: float = 0.01,
    max_candidate_count: int = 64,
    max_candidate_draft_tokens: int = 256,
) -> bool:
    """Add request-scoped D3 routes to a vLLM FastAPI application."""

    if not prefix.startswith("/") or prefix.endswith("/"):
        raise ValueError("route prefix must start with '/' and not end with '/'")
    if registration_wait_timeout < 0:
        raise ValueError("registration_wait_timeout must be non-negative")
    if registration_retry_interval <= 0:
        raise ValueError("registration_retry_interval must be positive")
    if max_candidate_count <= 0:
        raise ValueError("max_candidate_count must be positive")
    if max_candidate_draft_tokens <= 0:
        raise ValueError("max_candidate_draft_tokens must be positive")
    add_api_route = getattr(app, "add_api_route", None)
    if add_api_route is None or not callable(add_api_route):
        raise TypeError("app must provide add_api_route")
    state = getattr(app, "state", None)
    if state is None:
        raise TypeError("app must provide state")
    marker = "_self_speculation_http_routes"
    if getattr(state, marker, False):
        return False
    try:
        from fastapi import HTTPException
    except ImportError as error:  # pragma: no cover - optional server dependency
        raise ImportError(
            "vLLM HTTP routes require: pip install 'self-speculation[server]'"
        ) from error

    async def resolve_engine_client() -> Any:
        if engine_client_resolver is None:
            engine_client = getattr(app.state, "engine_client", None)
        else:
            engine_client = engine_client_resolver(app)
            if inspect.isawaitable(engine_client):
                engine_client = await engine_client
        if engine_client is None:
            raise HTTPException(status_code=503, detail="vLLM engine client unavailable")
        return engine_client

    async def feedback() -> VLLMCollectiveRPCDraftFeedback:
        engine_client = await resolve_engine_client()
        return VLLMCollectiveRPCDraftFeedback(engine_client, timeout=timeout)

    async def when_request_active(callback: Callable[[], Awaitable[Any]]) -> Any:
        deadline = asyncio.get_running_loop().time() + registration_wait_timeout
        while True:
            try:
                return await callback()
            except VLLMRequestNotActiveError:
                if asyncio.get_running_loop().time() >= deadline:
                    raise
                await asyncio.sleep(registration_retry_interval)

    def request_from_payload(payload: Mapping[str, Any]) -> DraftRequest:
        boundary_payload = payload.get("boundary")
        if not isinstance(boundary_payload, Mapping):
            raise ValueError("boundary must be an object")
        boundary = DraftBoundary(
            text=boundary_payload.get("text"),
            token_ids=tuple(
                int(token) for token in boundary_payload.get("token_ids") or ()
            ),
        )
        return DraftRequest(
            request_id=str(payload.get("request_id") or ""),
            text=str(payload.get("text") or ""),
            token_ids=tuple(int(token) for token in payload.get("token_ids") or ()),
            boundary=boundary,
            prompt_token_count=(
                int(payload["prompt_token_count"])
                if payload.get("prompt_token_count") is not None
                else None
            ),
            metadata=dict(payload.get("metadata") or {}),
        )

    async def submit(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            draft = request_from_payload(payload)
            current_feedback = await feedback()
            receipt = await when_request_active(
                lambda: current_feedback.submit(draft)
            )
        except (TypeError, ValueError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except VLLMRequestNotActiveError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except VLLMIntegrationError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        return receipt_payload(receipt)

    def receipt_payload(receipt: DraftReceipt) -> dict[str, Any]:
        return {
            "request_id": receipt.request_id,
            "registered": receipt.registered,
            "draft_token_count": receipt.draft_token_count,
            "accepted_token_count": receipt.accepted_token_count,
            "details": dict(receipt.details),
        }

    async def submit_bundle(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            request_id = str(payload.get("request_id") or "")
            raw_drafts = payload.get("drafts")
            if not request_id:
                raise ValueError("request_id must not be empty")
            if not isinstance(raw_drafts, list) or not raw_drafts:
                raise ValueError("drafts must be a non-empty array")
            drafts = tuple(
                request_from_payload({**raw_draft, "request_id": request_id})
                for raw_draft in raw_drafts
                if isinstance(raw_draft, Mapping)
            )
            if len(drafts) != len(raw_drafts):
                raise ValueError("every bundled draft must be an object")
            bundle = DraftBundle(
                request_id=request_id,
                drafts=drafts,
                metadata=dict(payload.get("metadata") or {}),
            )
            current_feedback = await feedback()
            receipt = await when_request_active(
                lambda: current_feedback.submit_bundle(bundle)
            )
        except (TypeError, ValueError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except VLLMRequestNotActiveError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except VLLMIntegrationError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        return receipt_payload(receipt)

    async def submit_candidates(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            engine_client = await resolve_engine_client()
            builder = CandidateBundleBuilder(
                tokenizer=lambda text: _engine_token_ids(engine_client, text),
                max_candidates=max_candidate_count,
                max_draft_tokens=max_candidate_draft_tokens,
            )
            bundle = await builder.build(payload)
            current_feedback = VLLMCollectiveRPCDraftFeedback(
                engine_client, timeout=timeout
            )
            receipt = await when_request_active(
                lambda: current_feedback.submit_bundle(bundle)
            )
        except (TypeError, ValueError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except VLLMRequestNotActiveError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except VLLMIntegrationError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        return receipt_payload(receipt)

    async def clear(payload: dict[str, Any]) -> dict[str, Any]:
        request_id = str(payload.get("request_id") or "")
        if not request_id:
            raise HTTPException(status_code=422, detail="request_id must not be empty")
        try:
            verification = await (await feedback()).clear(request_id)
        except VLLMIntegrationError as error:
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

    async def status() -> dict[str, Any]:
        try:
            results = await (await feedback()).status()
        except VLLMIntegrationError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        return {"status": "ok", "worker_results": results}

    add_api_route(prefix + "/drafts", submit, methods=["POST"])
    add_api_route(prefix + "/draft-bundles", submit_bundle, methods=["POST"])
    add_api_route(prefix + "/candidates", submit_candidates, methods=["POST"])
    add_api_route(prefix + "/clear", clear, methods=["POST"])
    add_api_route(prefix + "/status", status, methods=["GET"])
    setattr(state, marker, True)
    return True


class SelfSpeculationEndpointPlugin:
    """Official vLLM endpoint-plugin wrapper for the request-scoped routes."""

    name = "self_speculation"
    required_tasks = ("generate",)

    def attach_router(self, app: Any) -> None:
        install_vllm_http_routes(
            app,
            engine_client_resolver=lambda current_app: getattr(
                current_app.state,
                _PLUGIN_ENGINE_CLIENT_STATE,
                None,
            ),
        )

    async def init_state(
        self,
        engine_client: Any | None,
        state: Any,
        args: Any,
    ) -> None:
        del args
        setattr(state, _PLUGIN_ENGINE_CLIENT_STATE, engine_client)


def install_vllm_request_id_hook(runner_class: type[Any] | None = None) -> bool:
    """Route current vLLM batch request IDs into a custom proposer.

    Current vLLM custom proposers receive token rows but not the corresponding
    request IDs. This narrow wrapper supplies ``input_batch.req_ids`` immediately
    before vLLM invokes the proposer. Returns ``True`` when newly installed.
    """

    if runner_class is None:
        try:
            from vllm.v1.worker.gpu_model_runner import GPUModelRunner
        except ImportError as error:  # pragma: no cover - requires optional vLLM
            raise ImportError("VLLMBoundaryProposer requires vLLM V1") from error
        runner_class = GPUModelRunner

    if getattr(runner_class, _HOOK_MARKER, False):
        return False
    original = getattr(runner_class, "propose_draft_token_ids", None)
    if original is None or not callable(original):
        raise RuntimeError(
            "unsupported vLLM version: GPUModelRunner.propose_draft_token_ids "
            "was not found"
        )

    @wraps(original)
    def with_request_ids(runner: Any, *args: Any, **kwargs: Any) -> Any:
        drafter = getattr(runner, "drafter", None)
        setter = getattr(drafter, "set_request_ids", None)
        if setter is not None:
            input_batch = getattr(runner, "input_batch", None)
            request_ids = getattr(input_batch, "req_ids", None)
            if request_ids is None:
                raise RuntimeError("vLLM input_batch.req_ids is unavailable")
            setter(tuple(str(request_id) for request_id in request_ids))
        return original(runner, *args, **kwargs)

    setattr(runner_class, "propose_draft_token_ids", with_request_ids)
    setattr(runner_class, _HOOK_MARKER, True)
    return True


class VLLMBoundaryProposer:
    """vLLM ``custom_class`` proposer backed by ``BoundaryDraftStore``.

    Configure vLLM with this fully qualified class name and register drafts via
    the worker RPC bridge. Rows without a matching registered draft return no
    proposal, preserving ordinary target-model decoding.
    """

    def __init__(self, vllm_config: Any) -> None:
        speculative_config = getattr(vllm_config, "speculative_config", None)
        if speculative_config is None:
            raise ValueError("vLLM speculative_config is required")
        self.k = int(getattr(speculative_config, "num_speculative_tokens", 0))
        if self.k <= 0:
            raise ValueError("vLLM num_speculative_tokens must be positive")
        try:
            inject_window = int(
                os.environ.get("SELF_SPECULATION_INJECT_WINDOW", "200")
            )
        except ValueError as error:
            raise ValueError(
                "SELF_SPECULATION_INJECT_WINDOW must be an integer"
            ) from error
        self.store = BoundaryDraftStore(
            max_draft_tokens=self.k,
            inject_window=inject_window,
        )
        self._request_ids: tuple[str, ...] = ()
        self._alias_lock = threading.RLock()
        self._request_aliases: dict[str, str] = {}
        install_vllm_request_id_hook()
        install_vllm_worker_rpc()

    def set_request_ids(self, request_ids: tuple[str, ...]) -> None:
        self._request_ids = tuple(str(request_id) for request_id in request_ids)

    def register_draft(
        self,
        request_id: str,
        draft_token_ids: list[int],
        boundary_token_ids: list[int],
        prompt_token_count: int = 0,
        *,
        external_request_id: str | None = None,
    ) -> dict[str, Any]:
        external_id = external_request_id or request_id
        with self._alias_lock:
            receipt = self.store.register(
                DraftRequest(
                    request_id=request_id,
                    token_ids=tuple(draft_token_ids),
                    boundary=DraftBoundary(token_ids=tuple(boundary_token_ids)),
                    prompt_token_count=prompt_token_count,
                )
            )
            previous = self._request_aliases.get(external_id)
            if previous is not None and previous != request_id:
                self.store.clear(previous)
            self._request_aliases[external_id] = request_id
        return {
            "status": "ok",
            "request_id": external_id,
            "internal_request_id": receipt.request_id,
            "registered": receipt.registered,
            "draft_token_count": receipt.draft_token_count,
            **dict(receipt.details),
        }

    def register_draft_bundle(
        self,
        request_id: str,
        draft_token_ids: list[list[int]],
        boundary_token_ids: list[list[int]],
        prompt_token_count: int = 0,
        *,
        candidate_ids: list[str | None] | None = None,
        external_request_id: str | None = None,
    ) -> dict[str, Any]:
        if not draft_token_ids:
            raise ValueError("draft bundle must contain at least one candidate")
        if len(boundary_token_ids) != len(draft_token_ids):
            raise ValueError("every draft candidate needs boundary token IDs")
        if candidate_ids is not None and len(candidate_ids) != len(draft_token_ids):
            raise ValueError("candidate_ids must align with draft candidates")
        external_id = external_request_id or request_id
        drafts = tuple(
            DraftRequest(
                request_id=request_id,
                token_ids=tuple(tokens),
                boundary=DraftBoundary(token_ids=tuple(boundary_token_ids[index])),
                prompt_token_count=prompt_token_count,
                metadata={
                    "candidate_id": (
                        candidate_ids[index]
                        if candidate_ids is not None
                        else str(index)
                    )
                },
            )
            for index, tokens in enumerate(draft_token_ids)
        )
        with self._alias_lock:
            receipt = self.store.register_bundle(
                DraftBundle(request_id=request_id, drafts=drafts)
            )
            previous = self._request_aliases.get(external_id)
            if previous is not None and previous != request_id:
                self.store.clear(previous)
            self._request_aliases[external_id] = request_id
        return {
            "status": "ok",
            "request_id": external_id,
            "internal_request_id": receipt.request_id,
            "registered": receipt.registered,
            "draft_token_count": receipt.draft_token_count,
            **dict(receipt.details),
        }

    def clear_request(self, request_id: str) -> dict[str, Any]:
        with self._alias_lock:
            internal_request_id = self._request_aliases.pop(
                request_id, request_id
            )
        verification = self.store.take_outcome(internal_request_id)
        return {
            "status": "cleared",
            "request_id": request_id,
            "internal_request_id": internal_request_id,
            "removed": verification is not None,
            **(
                {"verification": verification.to_mapping()}
                if verification is not None
                else {}
            ),
        }

    def clear_all(self) -> dict[str, Any]:
        with self._alias_lock:
            self._request_aliases.clear()
            removed = self.store.clear_all()
        return {"status": "cleared", "removed": removed}

    def status(self) -> dict[str, Any]:
        return asdict(self.store.snapshot())

    def propose(
        self,
        sampled_token_ids: list[list[int]],
        num_tokens_no_spec: Any,
        token_ids_cpu: Any,
        slot_mappings: Any = None,
    ) -> list[list[int]]:
        del slot_mappings
        batch_size = len(sampled_token_ids)
        if len(self._request_ids) != batch_size:
            return [[] for _ in range(batch_size)]

        proposals: list[list[int]] = []
        for index, sampled_ids in enumerate(sampled_token_ids):
            if not sampled_ids:
                proposals.append([])
                continue
            sequence_length = int(num_tokens_no_spec[index])
            proposal = self.store.offer(
                self._request_ids[index],
                token_ids_cpu[index],
                sequence_length=sequence_length,
                max_tokens=self.k,
            )
            proposals.append(list(proposal.token_ids) if proposal else [])
        return proposals

    def load_model(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
