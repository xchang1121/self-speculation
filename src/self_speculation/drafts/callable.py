"""Adapt application callbacks to the portable D3 feedback protocol."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .base import DraftReceipt, DraftRequest, DraftVerificationOutcome


SubmitResult = DraftReceipt | Mapping[str, Any] | bool | None
DraftSubmitter = Callable[[DraftRequest], SubmitResult | Awaitable[SubmitResult]]
DraftClearer = Callable[[str], Any | Awaitable[Any]]


async def _invoke(callback: Callable[..., Any], *args: Any) -> Any:
    if inspect.iscoroutinefunction(callback):
        return await callback(*args)
    result = await asyncio.to_thread(callback, *args)
    if inspect.isawaitable(result):
        return await result
    return result


def _optional_int(value: Any, field: str) -> int | None:
    if value is None:
        return None
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{field} must be an integer or None") from error
    if result < 0:
        raise ValueError(f"{field} must be non-negative")
    return result


def _registered(payload: Mapping[str, Any]) -> bool:
    if "registered" in payload:
        return bool(payload["registered"])
    status = str(payload.get("status", "")).strip().lower()
    if status:
        return status in {"ok", "accepted", "registered", "success"}
    return True


def normalize_draft_receipt(
    result: SubmitResult,
    draft: DraftRequest,
) -> DraftReceipt:
    """Normalize common adapter return shapes and preserve raw details."""

    if isinstance(result, DraftReceipt):
        if result.request_id != draft.request_id:
            raise ValueError("draft receipt request_id does not match the request")
        return result
    if isinstance(result, Mapping):
        request_id = str(result.get("request_id") or draft.request_id)
        if request_id != draft.request_id:
            raise ValueError("draft receipt request_id does not match the request")
        draft_count = result.get(
            "draft_token_count",
            result.get("n_tokens", len(draft.token_ids) or None),
        )
        accepted_count = result.get(
            "accepted_token_count", result.get("accepted_tokens")
        )
        return DraftReceipt(
            request_id=request_id,
            registered=_registered(result),
            draft_token_count=_optional_int(draft_count, "draft_token_count"),
            accepted_token_count=_optional_int(
                accepted_count, "accepted_token_count"
            ),
            details=dict(result),
        )
    if result is None or isinstance(result, bool):
        return DraftReceipt(
            request_id=draft.request_id,
            registered=True if result is None else result,
            draft_token_count=len(draft.token_ids) or None,
        )
    raise TypeError(
        "draft submitter must return DraftReceipt, mapping, bool, or None"
    )


@dataclass(slots=True)
class CallableDraftFeedback:
    """Use sync or async callbacks as an engine's D3 side channel."""

    submitter: DraftSubmitter
    clearer: DraftClearer | None = None
    name: str = "callable-draft-feedback"

    async def submit(self, draft: DraftRequest) -> DraftReceipt:
        result = await _invoke(self.submitter, draft)
        return normalize_draft_receipt(result, draft)

    async def clear(
        self, request_id: str
    ) -> DraftVerificationOutcome | None:
        if self.clearer is None:
            return None
        result = await _invoke(self.clearer, request_id)
        if result is None or isinstance(result, bool):
            return None
        if isinstance(result, DraftVerificationOutcome):
            if result.request_id != request_id:
                raise ValueError(
                    "verification outcome request_id does not match the request"
                )
            return result
        if isinstance(result, Mapping):
            verification = result.get("verification", result)
            if isinstance(verification, Mapping):
                return DraftVerificationOutcome.from_mapping(
                    request_id,
                    verification,
                )
        raise TypeError(
            "draft clearer must return DraftVerificationOutcome, mapping, bool, or None"
        )
