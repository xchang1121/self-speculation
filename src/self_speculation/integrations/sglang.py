"""Request-scoped boundary drafts for SGLang's NGRAM verifier.

The integration is loaded through SGLang's official ``sglang.srt.plugins``
entry-point group.  It leaves ordinary NGRAM matching intact and overlays a
linear boundary draft only for the matching stable request ID.  SGLang then
verifies that candidate in its normal target-model speculative pass.
"""

from __future__ import annotations

import base64
import json
import os
import threading
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from ..drafts import BoundaryDraftStore, DraftBoundary, DraftBundle, DraftRequest
from ..drafts.http import HTTPDraftFeedback


SGLANG_CONTROL_PREFIX = "self-speculation:v1:"
_STORE_ATTRIBUTE = "_self_speculation_boundary_drafts"
_PENDING_ATTRIBUTE = "_self_speculation_pending_drafts"
_PENDING_LOCK_ATTRIBUTE = "_self_speculation_pending_drafts_lock"
_PLUGIN_REGISTERED = False


class SGLangIntegrationError(RuntimeError):
    pass


def _encode_control(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(
        {"version": 1, **dict(payload)},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    encoded = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
    return SGLANG_CONTROL_PREFIX + encoded


def _decode_control(corpus_id: str) -> Mapping[str, Any] | None:
    if not corpus_id.startswith(SGLANG_CONTROL_PREFIX):
        return None
    encoded = corpus_id[len(SGLANG_CONTROL_PREFIX) :]
    try:
        padding = "=" * (-len(encoded) % 4)
        value = json.loads(
            base64.urlsafe_b64decode(encoded + padding).decode("utf-8")
        )
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SGLangIntegrationError(
            "invalid self-speculation SGLang control corpus ID"
        ) from error
    if not isinstance(value, Mapping) or value.get("version") != 1:
        raise SGLangIntegrationError(
            "unsupported self-speculation SGLang control payload"
        )
    return value


def _positive_int(value: Any, field: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise SGLangIntegrationError(f"{field} must be an integer") from error
    if result <= 0:
        raise SGLangIntegrationError(f"{field} must be positive")
    return result


def _token_ids(value: Any, field: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise SGLangIntegrationError(f"{field} must be a token-ID sequence")
    result = tuple(int(token) for token in value)
    if not result:
        raise SGLangIntegrationError(f"{field} must not be empty")
    return result


def _store(worker: Any, *, required: bool = False) -> BoundaryDraftStore | None:
    value = getattr(worker, _STORE_ATTRIBUTE, None)
    if isinstance(value, BoundaryDraftStore):
        return value
    if required:
        raise SGLangIntegrationError(
            "SGLang boundary store is unavailable; start the server with "
            "--speculative-algorithm NGRAM and load the self_speculation plugin"
        )
    return None


def _worker_init(original: Callable[..., Any], worker: Any, *args: Any, **kwargs: Any) -> Any:
    result = original(worker, *args, **kwargs)
    draft_token_num = _positive_int(
        getattr(worker, "draft_token_num", None), "draft_token_num"
    )
    try:
        inject_window = int(os.environ.get("SELF_SPECULATION_INJECT_WINDOW", "200"))
    except ValueError as error:
        raise SGLangIntegrationError(
            "SELF_SPECULATION_INJECT_WINDOW must be an integer"
        ) from error
    setattr(
        worker,
        _STORE_ATTRIBUTE,
        BoundaryDraftStore(
            max_draft_tokens=draft_token_num,
            inject_window=inject_window,
        ),
    )
    setattr(worker, _PENDING_ATTRIBUTE, {})
    setattr(worker, _PENDING_LOCK_ATTRIBUTE, threading.RLock())
    return result


def _previous_tokens(worker: Any, batch: Any, index: int) -> list[int]:
    grammar_needs_sync = getattr(batch, "grammar_needs_sync", None)
    use_previous = bool(getattr(worker, "enable_overlap", False))
    if callable(grammar_needs_sync):
        use_previous = use_previous and not grammar_needs_sync()
    if not use_previous:
        return []
    lengths = getattr(worker, "prev_accept_lens", ())
    tokens = getattr(worker, "prev_token_ids", ())
    if index >= len(lengths):
        return []
    stride = int(worker.draft_token_num)
    accepted = int(lengths[index])
    return [
        int(token)
        for token in tokens[index * stride : index * stride + accepted]
    ]


def _prepare_draft_tokens(
    original: Callable[..., Any], worker: Any, batch: Any
) -> tuple[Any, Any]:
    original_result = original(worker, batch)
    store = _store(worker)
    if store is None:
        return original_result

    requests = list(getattr(batch, "reqs", ()))
    if not requests:
        return original_result
    pending = getattr(worker, _PENDING_ATTRIBUTE, {})
    pending_lock = getattr(worker, _PENDING_LOCK_ATTRIBUTE, None)
    resolved_pending: list[tuple[Any, Mapping[str, Any]]] = []
    if pending_lock is not None:
        with pending_lock:
            for request in requests:
                request_id = str(getattr(request, "rid", None) or "")
                payload = pending.pop(request_id, None)
                if payload is None:
                    continue
                resolved_pending.append((request, payload))
    for request, payload in resolved_pending:
        store.register_bundle(
            _control_bundle(payload, prompt_token_count=len(request.origin_input_ids))
        )
    if store.snapshot().active_requests == 0:
        return original_result
    draft_token_num = int(worker.draft_token_num)

    try:
        import numpy as np
    except ImportError as error:  # pragma: no cover - SGLang requires numpy
        raise SGLangIntegrationError("SGLang boundary injection requires numpy") from error

    draft_tokens, mask = original_result
    rows = np.array(draft_tokens, copy=True).reshape(len(requests), draft_token_num)
    trees = np.array(mask, copy=True).reshape(
        len(requests), draft_token_num, draft_token_num
    )
    linear_tree = np.tril(
        np.ones((draft_token_num, draft_token_num), dtype=trees.dtype)
    )
    changed = False
    for index, request in enumerate(requests):
        request_id = getattr(request, "rid", None)
        if not request_id:
            continue
        sequence = [int(token) for token in request.origin_input_ids]
        sequence.extend(int(token) for token in request.output_ids)
        sequence.extend(_previous_tokens(worker, batch, index))
        proposal = store.offer(
            str(request_id),
            sequence,
            sequence_length=len(sequence),
            max_tokens=draft_token_num,
        )
        if proposal is None:
            continue
        proposed = proposal.token_ids
        rows[index, : len(proposed)] = proposed
        if len(proposed) < draft_token_num:
            # SGLang's NGRAM verifier has a fixed node budget.  Padding remains
            # correctness-preserving because every node is target-verified.
            rows[index, len(proposed) :] = proposed[-1]
        trees[index] = linear_tree
        changed = True

    if not changed:
        return original_result
    return rows.reshape(-1), trees.reshape(-1)


def _control_bundle(
    payload: Mapping[str, Any], *, prompt_token_count: int | None = None
) -> DraftBundle:
    request_id = str(payload.get("request_id") or "")
    if not request_id:
        raise SGLangIntegrationError("request_id is required")
    raw_drafts = payload.get("drafts")
    if not isinstance(raw_drafts, list) or not raw_drafts:
        raise SGLangIntegrationError("drafts must be a non-empty array")
    drafts = []
    for raw in raw_drafts:
        if not isinstance(raw, Mapping):
            raise SGLangIntegrationError("every draft must be an object")
        metadata = raw.get("metadata") or {}
        if not isinstance(metadata, Mapping):
            raise SGLangIntegrationError("draft metadata must be an object")
        observed_prompt_count = raw.get("prompt_token_count")
        drafts.append(
            DraftRequest(
                request_id=request_id,
                token_ids=_token_ids(raw.get("token_ids"), "token_ids"),
                boundary=DraftBoundary(
                    token_ids=_token_ids(
                        raw.get("boundary_token_ids"), "boundary_token_ids"
                    )
                ),
                prompt_token_count=(
                    prompt_token_count
                    if observed_prompt_count is None
                    else int(observed_prompt_count)
                ),
                metadata=dict(metadata),
            )
        )
    return DraftBundle(request_id=request_id, drafts=tuple(drafts))


def _register_control(worker: Any, payload: Mapping[str, Any]) -> int:
    if payload.get("op") != "replace":
        raise SGLangIntegrationError("expected a replace control payload")
    bundle = _control_bundle(payload)
    store = _store(worker, required=True)
    if any(draft.prompt_token_count is None for draft in bundle.drafts):
        pending = getattr(worker, _PENDING_ATTRIBUTE, None)
        pending_lock = getattr(worker, _PENDING_LOCK_ATTRIBUTE, None)
        if not isinstance(pending, dict) or pending_lock is None:
            raise SGLangIntegrationError("SGLang pending draft state is unavailable")
        store.clear(bundle.request_id)
        with pending_lock:
            pending[bundle.request_id] = dict(payload)
        return min(
            max(len(draft.token_ids) for draft in bundle.drafts),
            store.max_draft_tokens,
        )
    pending = getattr(worker, _PENDING_ATTRIBUTE, None)
    pending_lock = getattr(worker, _PENDING_LOCK_ATTRIBUTE, None)
    if isinstance(pending, dict) and pending_lock is not None:
        with pending_lock:
            pending.pop(bundle.request_id, None)
    receipt = store.register_bundle(bundle)
    return int(receipt.draft_token_count or 0)


def _add_external_corpus(
    original: Callable[..., Any],
    worker: Any,
    corpus_id: str,
    token_chunks: list[list[int]],
) -> int:
    control = _decode_control(corpus_id)
    if control is None:
        return original(worker, corpus_id, token_chunks)
    return _register_control(worker, control)


def _commit_corpus_load(
    original: Callable[..., Any],
    worker: Any,
    corpus_id: str,
    loaded_token_count: int,
) -> Any:
    if _decode_control(corpus_id) is not None:
        return None
    return original(worker, corpus_id, loaded_token_count)


def _remove_external_corpus(
    original: Callable[..., Any], worker: Any, corpus_id: str
) -> Any:
    control = _decode_control(corpus_id)
    if control is None:
        return original(worker, corpus_id)
    if control.get("op") != "clear":
        raise SGLangIntegrationError("expected a clear control payload")
    request_id = str(control.get("request_id") or "")
    if not request_id:
        raise SGLangIntegrationError("request_id is required")
    pending = getattr(worker, _PENDING_ATTRIBUTE, None)
    pending_lock = getattr(worker, _PENDING_LOCK_ATTRIBUTE, None)
    if isinstance(pending, dict) and pending_lock is not None:
        with pending_lock:
            pending.pop(request_id, None)
    _store(worker, required=True).clear(request_id)
    return None


def _clear_cache_pool(original: Callable[..., Any], worker: Any) -> Any:
    result = original(worker)
    store = _store(worker)
    if store is not None:
        store.clear_all()
    pending = getattr(worker, _PENDING_ATTRIBUTE, None)
    pending_lock = getattr(worker, _PENDING_LOCK_ATTRIBUTE, None)
    if isinstance(pending, dict) and pending_lock is not None:
        with pending_lock:
            pending.clear()
    return result


def _external_corpus_manager_add(
    original: Callable[..., Any],
    manager: Any,
    request: Any,
    *,
    _output_type: Any = None,
) -> Any:
    """Register tiny control payloads without occupying the corpus load slot."""

    corpus_id = str(getattr(request, "corpus_id", "") or "")
    if _decode_control(corpus_id) is None:
        return original(manager, request)
    if _output_type is None:
        from sglang.srt.managers.io_struct import AddExternalCorpusReqOutput

        _output_type = AddExternalCorpusReqOutput
    try:
        loaded = manager._worker.add_external_corpus(
            corpus_id, list(getattr(request, "token_chunks", None) or ())
        )
        manager._worker.commit_corpus_load(corpus_id, loaded)
        return _output_type(
            success=True,
            corpus_id=corpus_id,
            message="Registered request-scoped self-speculation draft.",
            loaded_token_count=loaded,
        )
    except Exception as error:
        return _output_type(success=False, message=str(error))


_HOOKS = (
    ("sglang.srt.speculative.ngram_worker.NGRAMWorker.__init__", _worker_init),
    (
        "sglang.srt.speculative.ngram_worker.NGRAMWorker._prepare_draft_tokens",
        _prepare_draft_tokens,
    ),
    (
        "sglang.srt.speculative.ngram_worker.NGRAMWorker.add_external_corpus",
        _add_external_corpus,
    ),
    (
        "sglang.srt.speculative.ngram_worker.NGRAMWorker.commit_corpus_load",
        _commit_corpus_load,
    ),
    (
        "sglang.srt.speculative.ngram_worker.NGRAMWorker.remove_external_corpus",
        _remove_external_corpus,
    ),
    (
        "sglang.srt.speculative.ngram_worker.NGRAMWorker.clear_cache_pool",
        _clear_cache_pool,
    ),
    (
        "sglang.srt.speculative.external_corpus_manager.ExternalCorpusManager.add",
        _external_corpus_manager_add,
    ),
)


def _register_hooks(registry: Any, around: Any) -> None:
    for target, hook in _HOOKS:
        registry.register(target, hook, around)


def install_sglang_plugin() -> bool:
    """Register hooks through SGLang's official general-plugin interface."""

    global _PLUGIN_REGISTERED
    if _PLUGIN_REGISTERED:
        return False
    try:
        from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    except ImportError as error:  # pragma: no cover - optional SGLang dependency
        raise ImportError(
            "SGLang D3 integration requires a build with sglang.srt.plugins"
        ) from error
    _register_hooks(HookRegistry, HookType.AROUND)
    _PLUGIN_REGISTERED = True
    return True


class SGLangHTTPDraftFeedback(HTTPDraftFeedback):
    """Use SGLang's NGRAM corpus routes as a request-scoped control plane."""

    def __init__(self, base_url: str, **kwargs: Any) -> None:
        kwargs.setdefault("submit_path", "/add_external_corpus")
        kwargs.setdefault("clear_path", "/remove_external_corpus")
        kwargs.setdefault("clear_method", "POST")
        kwargs.setdefault("name", "sglang-http-draft-feedback")
        super().__init__(base_url, **kwargs)

    def submit_payload(self, bundle: DraftBundle) -> Mapping[str, Any]:
        for draft in bundle.drafts:
            if not draft.token_ids:
                raise ValueError("SGLang draft feedback requires token_ids")
            if draft.boundary is None or not draft.boundary.token_ids:
                raise ValueError("SGLang draft feedback requires boundary token_ids")
        return {
            "corpus_id": _encode_control(
                {
                    "op": "replace",
                    "request_id": bundle.request_id,
                    "drafts": [
                        {
                            "token_ids": list(draft.token_ids),
                            "boundary_token_ids": list(
                                draft.boundary.token_ids
                                if draft.boundary is not None
                                else ()
                            ),
                            "prompt_token_count": draft.prompt_token_count,
                            "metadata": dict(draft.metadata),
                        }
                        for draft in bundle.drafts
                    ],
                }
            ),
            # The native route currently requires documents even though the
            # plugin transports exact IDs in corpus_id.  This token is ignored.
            "documents": ["_"],
        }

    def clear_payload(self, request_id: str) -> Mapping[str, Any]:
        return {
            "corpus_id": _encode_control(
                {"op": "clear", "request_id": request_id}
            )
        }
