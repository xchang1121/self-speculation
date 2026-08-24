"""Thread-safe, request-scoped draft storage for engine-side verification."""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .base import DraftReceipt, DraftRequest


BoundaryTokenizer = Callable[[str], Sequence[int]]


def _last_subsequence(sequence: Sequence[int], needle: Sequence[int]) -> int | None:
    if not needle or len(needle) > len(sequence):
        return None
    for index in range(len(sequence) - len(needle), -1, -1):
        if tuple(sequence[index : index + len(needle)]) == tuple(needle):
            return index
    return None


def _common_prefix_length(left: Sequence[int], right: Sequence[int]) -> int:
    length = 0
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        length += 1
    return length


@dataclass(frozen=True, slots=True)
class DraftProposal:
    request_id: str
    token_ids: tuple[int, ...]
    skipped_prefix_tokens: int
    generated_body_tokens: int
    boundary_index: int


@dataclass(frozen=True, slots=True)
class DraftStoreSnapshot:
    active_requests: int
    registrations: int
    injections: int
    proposed_tokens: int
    divergent_drafts: int
    stale_drafts: int


@dataclass(slots=True)
class _RequestDraft:
    token_ids: tuple[int, ...]
    boundary_token_ids: tuple[int, ...]
    prompt_token_count: int
    fired: bool = False


class BoundaryDraftStore:
    """Offer a registered continuation once a generated boundary appears.

    ``offer`` accepts the full current sequence for one stable request ID. It
    excludes prompt tokens, locates the last boundary, verifies that tokens
    already generated after the boundary are a prefix of the registered draft,
    and returns only the ungenerated suffix.
    """

    def __init__(
        self,
        *,
        max_draft_tokens: int = 20,
        inject_window: int = 200,
        boundary_tokenizer: BoundaryTokenizer | None = None,
    ) -> None:
        if max_draft_tokens <= 0:
            raise ValueError("max_draft_tokens must be positive")
        if inject_window < 0:
            raise ValueError("inject_window must be non-negative")
        self.max_draft_tokens = max_draft_tokens
        self.inject_window = inject_window
        self.boundary_tokenizer = boundary_tokenizer
        self._lock = threading.RLock()
        self._requests: dict[str, _RequestDraft] = {}
        self._registrations = 0
        self._injections = 0
        self._proposed_tokens = 0
        self._divergent_drafts = 0
        self._stale_drafts = 0

    def _boundary_tokens(self, draft: DraftRequest) -> tuple[int, ...]:
        if draft.boundary is None:
            raise ValueError("engine-side draft feedback requires a boundary")
        if draft.boundary.token_ids:
            return draft.boundary.token_ids
        if draft.boundary.text and self.boundary_tokenizer is not None:
            tokens = tuple(int(token) for token in self.boundary_tokenizer(
                draft.boundary.text
            ))
            if tokens:
                return tokens
        raise ValueError(
            "draft boundary needs token_ids or a configured boundary_tokenizer"
        )

    def register(self, draft: DraftRequest) -> DraftReceipt:
        if not draft.token_ids:
            raise ValueError("engine-side draft feedback requires token_ids")
        boundary_tokens = self._boundary_tokens(draft)
        if draft.prompt_token_count is None:
            raise ValueError(
                "engine-side draft feedback requires prompt_token_count"
            )
        token_ids = draft.token_ids[: self.max_draft_tokens]
        prompt_token_count = draft.prompt_token_count
        with self._lock:
            replaced = draft.request_id in self._requests
            self._requests[draft.request_id] = _RequestDraft(
                token_ids=token_ids,
                boundary_token_ids=boundary_tokens,
                prompt_token_count=prompt_token_count,
            )
            self._registrations += 1
        return DraftReceipt(
            request_id=draft.request_id,
            registered=True,
            draft_token_count=len(token_ids),
            details={
                "boundary_token_count": len(boundary_tokens),
                "replaced": replaced,
            },
        )

    def offer(
        self,
        request_id: str,
        sequence_token_ids: Sequence[int],
        *,
        sequence_length: int | None = None,
        max_tokens: int | None = None,
    ) -> DraftProposal | None:
        if sequence_length is None:
            sequence_length = len(sequence_token_ids)
        if sequence_length < 0 or sequence_length > len(sequence_token_ids):
            raise ValueError("sequence_length is outside sequence_token_ids")
        if max_tokens is not None and max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

        with self._lock:
            state = self._requests.get(request_id)
            if state is None or state.fired:
                return None
            generated = tuple(
                int(token)
                for token in sequence_token_ids[
                    min(state.prompt_token_count, sequence_length) : sequence_length
                ]
            )
            boundary_index = _last_subsequence(
                generated, state.boundary_token_ids
            )
            if boundary_index is None:
                return None

            body_start = boundary_index + len(state.boundary_token_ids)
            generated_body = generated[body_start:]
            if len(generated_body) > self.inject_window:
                state.fired = True
                self._stale_drafts += 1
                return None

            common = _common_prefix_length(generated_body, state.token_ids)
            if common != len(generated_body):
                state.fired = True
                self._divergent_drafts += 1
                return None

            remaining = state.token_ids[common:]
            if not remaining:
                state.fired = True
                return None
            limit = min(max_tokens or self.max_draft_tokens, self.max_draft_tokens)
            proposed = remaining[:limit]
            state.fired = True
            self._injections += 1
            self._proposed_tokens += len(proposed)
            return DraftProposal(
                request_id=request_id,
                token_ids=proposed,
                skipped_prefix_tokens=common,
                generated_body_tokens=len(generated_body),
                boundary_index=boundary_index,
            )

    def clear(self, request_id: str) -> bool:
        with self._lock:
            return self._requests.pop(request_id, None) is not None

    def clear_all(self) -> int:
        with self._lock:
            count = len(self._requests)
            self._requests.clear()
            return count

    def snapshot(self) -> DraftStoreSnapshot:
        with self._lock:
            return DraftStoreSnapshot(
                active_requests=len(self._requests),
                registrations=self._registrations,
                injections=self._injections,
                proposed_tokens=self._proposed_tokens,
                divergent_drafts=self._divergent_drafts,
                stale_drafts=self._stale_drafts,
            )


@dataclass(slots=True)
class BoundaryDraftFeedback:
    """Expose ``BoundaryDraftStore`` through the controller feedback protocol."""

    store: BoundaryDraftStore
    name: str = "boundary-draft-store"

    async def submit(self, draft: DraftRequest) -> DraftReceipt:
        return self.store.register(draft)

    async def clear(self, request_id: str) -> None:
        self.store.clear(request_id)
