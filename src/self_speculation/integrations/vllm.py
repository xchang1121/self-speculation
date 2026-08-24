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

from ..drafts import (
    BoundaryDraftStore,
    DraftBoundary,
    DraftReceipt,
    DraftRequest,
    HTTPDraftFeedback,
)


_HOOK_MARKER = "_self_speculation_request_id_hook"
_RPC_MARKER = "_self_speculation_worker_rpc"
REGISTER_DRAFT_RPC = "self_speculation_register_draft"
CLEAR_DRAFT_RPC = "self_speculation_clear_draft"
DRAFT_STATUS_RPC = "self_speculation_draft_status"
EngineClientResolver = Callable[[Any], Any | Awaitable[Any]]
_PLUGIN_ENGINE_CLIENT_STATE = "_self_speculation_engine_client"


class VLLMIntegrationError(RuntimeError):
    pass


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
        raise VLLMIntegrationError(
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
    method_names = (REGISTER_DRAFT_RPC, CLEAR_DRAFT_RPC, DRAFT_STATUS_RPC)
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
        registered = [
            result
            for result in results
            if str(result.get("status", "")).lower() in {"ok", "registered"}
            and bool(result.get("registered", True))
        ]
        if not registered:
            raise VLLMIntegrationError(
                "no vLLM worker has VLLMBoundaryProposer installed"
            )
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

    async def clear(self, request_id: str) -> None:
        await self._rpc(CLEAR_DRAFT_RPC, (request_id,))

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
) -> bool:
    """Add request-scoped D3 routes to a vLLM FastAPI application."""

    if not prefix.startswith("/") or prefix.endswith("/"):
        raise ValueError("route prefix must start with '/' and not end with '/'")
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

    async def feedback() -> VLLMCollectiveRPCDraftFeedback:
        if engine_client_resolver is None:
            engine_client = getattr(app.state, "engine_client", None)
        else:
            engine_client = engine_client_resolver(app)
            if inspect.isawaitable(engine_client):
                engine_client = await engine_client
        if engine_client is None:
            raise HTTPException(status_code=503, detail="vLLM engine client unavailable")
        return VLLMCollectiveRPCDraftFeedback(engine_client, timeout=timeout)

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
            receipt = await (await feedback()).submit(draft)
        except (TypeError, ValueError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except VLLMIntegrationError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        return {
            "request_id": receipt.request_id,
            "registered": receipt.registered,
            "draft_token_count": receipt.draft_token_count,
            "accepted_token_count": receipt.accepted_token_count,
            "details": dict(receipt.details),
        }

    async def clear(payload: dict[str, Any]) -> dict[str, Any]:
        request_id = str(payload.get("request_id") or "")
        if not request_id:
            raise HTTPException(status_code=422, detail="request_id must not be empty")
        try:
            await (await feedback()).clear(request_id)
        except VLLMIntegrationError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        return {"status": "cleared", "request_id": request_id}

    async def status() -> dict[str, Any]:
        try:
            results = await (await feedback()).status()
        except VLLMIntegrationError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        return {"status": "ok", "worker_results": results}

    add_api_route(prefix + "/drafts", submit, methods=["POST"])
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

    def clear_request(self, request_id: str) -> dict[str, Any]:
        with self._alias_lock:
            internal_request_id = self._request_aliases.pop(
                request_id, request_id
            )
        return {
            "status": "cleared",
            "request_id": request_id,
            "internal_request_id": internal_request_id,
            "removed": self.store.clear(internal_request_id),
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
