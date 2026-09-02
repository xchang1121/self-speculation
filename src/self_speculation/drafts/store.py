"""Thread-safe, request-scoped draft storage for engine-side verification."""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from .base import (
    DraftBundle,
    DraftReceipt,
    DraftRequest,
    DraftVerificationOutcome,
    DraftVerificationStep,
)


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


def _metadata_strings(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))


@dataclass(frozen=True, slots=True)
class DraftProposal:
    request_id: str
    token_ids: tuple[int, ...]
    skipped_prefix_tokens: int
    generated_body_tokens: int
    boundary_index: int
    candidate_index: int = 0
    candidate_count: int = 1
    candidate_id: str | None = None


@dataclass(frozen=True, slots=True)
class DraftStoreSnapshot:
    active_requests: int
    registrations: int
    injections: int
    proposed_tokens: int
    divergent_drafts: int
    stale_drafts: int
    registered_candidates: int = 0
    fallback_injections: int = 0
    resolved_proposals: int = 0
    unresolved_proposals: int = 0
    unresolved_draft_tokens: int = 0
    accepted_draft_tokens: int = 0
    rejected_draft_tokens: int = 0


@dataclass(slots=True)
class _CandidateDraft:
    token_ids: tuple[int, ...]
    boundary_token_ids: tuple[int, ...]
    prompt_token_count: int
    identity: tuple[object, ...]
    candidate_id: str | None = None
    candidate_ids: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()
    fired: bool = False


@dataclass(frozen=True, slots=True)
class _PendingProposal:
    token_ids: tuple[int, ...]
    sequence_length: int
    candidate_index: int
    candidate_id: str | None
    candidate_ids: tuple[str, ...]
    sources: tuple[str, ...]


@dataclass(slots=True)
class _RequestDraft:
    candidates: tuple[_CandidateDraft, ...]
    last_offer_sequence_length: int | None = None
    pending: _PendingProposal | None = None
    verification_steps: list[DraftVerificationStep] = field(default_factory=list)


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
        max_draft_tokens: int = 28,
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
        self._registered_candidates = 0
        self._fallback_injections = 0
        self._resolved_proposals = 0
        self._unresolved_proposals = 0
        self._unresolved_draft_tokens = 0
        self._accepted_draft_tokens = 0
        self._rejected_draft_tokens = 0

    @staticmethod
    def _request(
        candidates: tuple[_CandidateDraft, ...],
        *,
        previous: _RequestDraft | None = None,
    ) -> _RequestDraft:
        return _RequestDraft(
            candidates=candidates,
            last_offer_sequence_length=(
                previous.last_offer_sequence_length
                if previous is not None
                else None
            ),
            pending=previous.pending if previous is not None else None,
            verification_steps=(
                list(previous.verification_steps) if previous is not None else []
            ),
        )

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

    def _candidate(self, draft: DraftRequest) -> _CandidateDraft:
        if not draft.token_ids:
            raise ValueError("engine-side draft feedback requires token_ids")
        boundary_tokens = self._boundary_tokens(draft)
        if draft.prompt_token_count is None:
            raise ValueError(
                "engine-side draft feedback requires prompt_token_count"
            )
        token_ids = tuple(draft.token_ids[: self.max_draft_tokens])
        candidate_id_value = draft.metadata.get("candidate_id")
        candidate_id = (
            str(candidate_id_value).strip()
            if candidate_id_value is not None
            else None
        )
        if candidate_id == "":
            candidate_id = None
        candidate_ids = tuple(
            dict.fromkeys(
                value
                for value in (
                    *((candidate_id,) if candidate_id is not None else ()),
                    *_metadata_strings(draft.metadata.get("candidate_ids")),
                )
            )
        )
        sources = _metadata_strings(draft.metadata.get("sources"))
        identity: tuple[object, ...] = (
            ("id", candidate_id)
            if candidate_id is not None
            else ("tokens", token_ids, boundary_tokens)
        )
        return _CandidateDraft(
            token_ids=token_ids,
            boundary_token_ids=boundary_tokens,
            prompt_token_count=draft.prompt_token_count,
            identity=identity,
            candidate_id=candidate_id,
            candidate_ids=candidate_ids,
            sources=sources,
        )

    def register_bundle(self, bundle: DraftBundle) -> DraftReceipt:
        candidates: list[_CandidateDraft] = []
        seen: set[tuple[object, ...]] = set()
        for draft in bundle.drafts:
            candidate = self._candidate(draft)
            if candidate.identity in seen:
                continue
            seen.add(candidate.identity)
            candidates.append(candidate)
        if not candidates:
            raise ValueError("draft bundle has no distinct candidates")

        with self._lock:
            previous = self._requests.get(bundle.request_id)
            if previous is not None:
                previous_candidates = {
                    candidate.identity: candidate
                    for candidate in previous.candidates
                }
                for candidate in candidates:
                    old = previous_candidates.get(candidate.identity)
                    if old is not None:
                        candidate.fired = old.fired
            self._requests[bundle.request_id] = self._request(
                tuple(candidates),
                previous=previous,
            )
            self._registrations += 1
            self._registered_candidates += len(candidates)
        return DraftReceipt(
            request_id=bundle.request_id,
            registered=True,
            draft_token_count=max(len(candidate.token_ids) for candidate in candidates),
            details={
                "candidate_count": len(candidates),
                "input_candidate_count": len(bundle.drafts),
                "replaced": previous is not None,
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
            if state is None:
                return None
            self._resolve_from_sequence(
                state,
                sequence_token_ids,
                sequence_length,
            )
            if state.last_offer_sequence_length == sequence_length:
                return None
            for candidate_index, candidate in enumerate(state.candidates):
                if candidate.fired:
                    continue
                generated = tuple(
                    int(token)
                    for token in sequence_token_ids[
                        min(candidate.prompt_token_count, sequence_length) : sequence_length
                    ]
                )
                boundary_index = _last_subsequence(
                    generated, candidate.boundary_token_ids
                )
                if boundary_index is None:
                    continue

                body_start = boundary_index + len(candidate.boundary_token_ids)
                generated_body = generated[body_start:]
                if len(generated_body) > self.inject_window:
                    candidate.fired = True
                    self._stale_drafts += 1
                    continue

                common = _common_prefix_length(generated_body, candidate.token_ids)
                if common != len(generated_body):
                    candidate.fired = True
                    self._divergent_drafts += 1
                    continue

                remaining = candidate.token_ids[common:]
                if not remaining:
                    candidate.fired = True
                    continue
                limit = min(max_tokens or self.max_draft_tokens, self.max_draft_tokens)
                proposed = remaining[:limit]
                candidate.fired = True
                state.last_offer_sequence_length = sequence_length
                state.pending = _PendingProposal(
                    token_ids=tuple(proposed),
                    sequence_length=sequence_length,
                    candidate_index=candidate_index,
                    candidate_id=candidate.candidate_id,
                    candidate_ids=candidate.candidate_ids,
                    sources=candidate.sources,
                )
                self._injections += 1
                self._proposed_tokens += len(proposed)
                if candidate_index > 0:
                    self._fallback_injections += 1
                return DraftProposal(
                    request_id=request_id,
                    token_ids=proposed,
                    skipped_prefix_tokens=common,
                    generated_body_tokens=len(generated_body),
                    boundary_index=boundary_index,
                    candidate_index=candidate_index,
                    candidate_count=len(state.candidates),
                    candidate_id=candidate.candidate_id,
                )
            return None

    def _resolve_pending(
        self,
        state: _RequestDraft,
        accepted_tokens: int,
    ) -> bool:
        pending = state.pending
        if pending is None:
            return False
        if accepted_tokens < 0 or accepted_tokens > len(pending.token_ids):
            raise ValueError("accepted_tokens is outside the pending proposal")
        step = DraftVerificationStep(
            drafted_tokens=len(pending.token_ids),
            accepted_tokens=accepted_tokens,
            candidate_index=pending.candidate_index,
            candidate_id=pending.candidate_id,
            candidate_ids=pending.candidate_ids,
            sources=pending.sources,
        )
        state.verification_steps.append(step)
        state.pending = None
        self._resolved_proposals += 1
        self._accepted_draft_tokens += step.accepted_tokens
        self._rejected_draft_tokens += step.rejected_tokens
        return True

    def _resolve_from_sequence(
        self,
        state: _RequestDraft,
        sequence_token_ids: Sequence[int],
        sequence_length: int,
    ) -> None:
        pending = state.pending
        if pending is None or sequence_length <= pending.sequence_length:
            return
        emitted = tuple(
            int(token)
            for token in sequence_token_ids[
                pending.sequence_length : sequence_length
            ]
        )
        self._resolve_pending(
            state,
            _common_prefix_length(pending.token_ids, emitted),
        )

    def observe_acceptance(
        self,
        request_id: str,
        accepted_tokens: int,
    ) -> bool:
        """Resolve the most recent proposal from an engine verification callback."""

        with self._lock:
            state = self._requests.get(request_id)
            return (
                self._resolve_pending(state, int(accepted_tokens))
                if state is not None
                else False
            )

    @staticmethod
    def _outcome(
        request_id: str,
        state: _RequestDraft,
    ) -> DraftVerificationOutcome:
        pending = state.pending
        return DraftVerificationOutcome(
            request_id=request_id,
            steps=tuple(state.verification_steps),
            unresolved_proposals=1 if pending is not None else 0,
            unresolved_draft_tokens=(
                len(pending.token_ids) if pending is not None else 0
            ),
        )

    def outcome(self, request_id: str) -> DraftVerificationOutcome | None:
        with self._lock:
            state = self._requests.get(request_id)
            return (
                self._outcome(request_id, state)
                if state is not None
                else None
            )

    def take_outcome(self, request_id: str) -> DraftVerificationOutcome | None:
        with self._lock:
            state = self._requests.pop(request_id, None)
            if state is None:
                return None
            outcome = self._outcome(request_id, state)
            self._unresolved_proposals += outcome.unresolved_proposals
            self._unresolved_draft_tokens += outcome.unresolved_draft_tokens
            return outcome

    def clear(self, request_id: str) -> bool:
        return self.take_outcome(request_id) is not None

    def clear_all(self) -> int:
        with self._lock:
            count = len(self._requests)
            self._unresolved_proposals += sum(
                state.pending is not None for state in self._requests.values()
            )
            self._unresolved_draft_tokens += sum(
                len(state.pending.token_ids)
                for state in self._requests.values()
                if state.pending is not None
            )
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
                registered_candidates=self._registered_candidates,
                fallback_injections=self._fallback_injections,
                resolved_proposals=self._resolved_proposals,
                unresolved_proposals=self._unresolved_proposals,
                unresolved_draft_tokens=self._unresolved_draft_tokens,
                accepted_draft_tokens=self._accepted_draft_tokens,
                rejected_draft_tokens=self._rejected_draft_tokens,
            )


@dataclass(slots=True)
class BoundaryDraftFeedback:
    """Expose ``BoundaryDraftStore`` through the controller feedback protocol."""

    store: BoundaryDraftStore
    name: str = "boundary-draft-store"

    async def submit(self, bundle: DraftBundle) -> DraftReceipt:
        return self.store.register_bundle(bundle)

    async def clear(
        self, request_id: str
    ) -> DraftVerificationOutcome | None:
        return self.store.take_outcome(request_id)
