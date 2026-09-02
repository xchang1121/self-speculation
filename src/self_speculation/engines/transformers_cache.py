"""Request-owned immutable prefix states for Transformers forks."""

from __future__ import annotations

import copy
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class TransformersPrefixState:
    request_id: str
    token_ids: tuple[int, ...]
    past_key_values: Any


class TransformersPrefixCache:
    """Small shared LRU of KV states bound to an explicit model identity.

    Pass the same cache to Actor and fork engine instances only when their
    architecture, weights, tokenizer, device layout, and numeric type match.
    Returned KV state is deep-copied because Transformers caches are mutable.
    """

    def __init__(self, model_identity: str, *, max_entries: int = 8) -> None:
        if not model_identity.strip():
            raise ValueError("model_identity must not be empty")
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self.model_identity = model_identity
        self.max_entries = max_entries
        self._entries: OrderedDict[
            tuple[str, tuple[int, ...]], TransformersPrefixState
        ] = OrderedDict()
        self._lock = threading.RLock()

    def put(
        self,
        request_id: str,
        token_ids: tuple[int, ...],
        past_key_values: Any,
    ) -> None:
        if not request_id.strip() or not token_ids:
            raise ValueError("a prefix state needs a request ID and token IDs")
        key = (request_id, token_ids)
        with self._lock:
            self._entries[key] = TransformersPrefixState(
                request_id,
                token_ids,
                past_key_values,
            )
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def fork(
        self,
        request_id: str,
        token_ids: tuple[int, ...],
    ) -> TransformersPrefixState | None:
        with self._lock:
            match = max(
                (
                    (key, entry)
                    for key, entry in self._entries.items()
                    if entry.request_id == request_id
                    and token_ids[: len(entry.token_ids)] == entry.token_ids
                ),
                key=lambda item: len(item[1].token_ids),
                default=None,
            )
            if match is None:
                return None
            key, entry = match
            self._entries.move_to_end(key)
            return TransformersPrefixState(
                entry.request_id,
                entry.token_ids,
                copy.deepcopy(entry.past_key_values),
            )

    def clear(self, request_id: str | None = None) -> None:
        with self._lock:
            if request_id is None:
                self._entries.clear()
                return
            for key in tuple(self._entries):
                if key[0] == request_id:
                    del self._entries[key]

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


__all__ = ["TransformersPrefixCache", "TransformersPrefixState"]
