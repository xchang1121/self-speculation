"""vLLM custom proposer integration with stable request-ID routing.

The module deliberately imports vLLM only when installation is requested, so
the base package remains usable with other engines and on non-CUDA hosts.
"""

from __future__ import annotations

import os
from dataclasses import asdict
from functools import wraps
from typing import Any

from ..drafts import BoundaryDraftStore, DraftBoundary, DraftRequest


_HOOK_MARKER = "_self_speculation_request_id_hook"


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
        install_vllm_request_id_hook()

    def set_request_ids(self, request_ids: tuple[str, ...]) -> None:
        self._request_ids = tuple(str(request_id) for request_id in request_ids)

    def register_draft(
        self,
        request_id: str,
        draft_token_ids: list[int],
        boundary_token_ids: list[int],
        prompt_token_count: int = 0,
    ) -> dict[str, Any]:
        receipt = self.store.register(
            DraftRequest(
                request_id=request_id,
                token_ids=tuple(draft_token_ids),
                boundary=DraftBoundary(token_ids=tuple(boundary_token_ids)),
                prompt_token_count=prompt_token_count,
            )
        )
        return {
            "status": "ok",
            "request_id": receipt.request_id,
            "registered": receipt.registered,
            "draft_token_count": receipt.draft_token_count,
            **dict(receipt.details),
        }

    def clear_request(self, request_id: str) -> dict[str, Any]:
        return {
            "status": "cleared",
            "request_id": request_id,
            "removed": self.store.clear(request_id),
        }

    def clear_all(self) -> dict[str, Any]:
        return {"status": "cleared", "removed": self.store.clear_all()}

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
