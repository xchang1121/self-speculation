"""Record and analyze a faithful SPORK D1/D2 think-mode replay.

Raw prompts and generations are written only to the caller-selected recording
path.  The checked-in report should contain hashes and aggregate metrics only.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import os
import re
import tempfile
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

if __package__:
    from .d3_tape_ablation import parse_tape
else:
    from d3_tape_ablation import parse_tape  # type: ignore[no-redef]


TOOL_CALL_RE = re.compile(r"<tool_call>\s*", re.DOTALL)
THINK_END_PREFIX = '</think>\n\n<tool_call>\n{"name": "'
DEFAULT_TOKENIZER = "Qwen/Qwen3-1.7B"
DEFAULT_TOKENIZER_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _request_hash(messages: Any, tools: Any) -> str:
    return _sha256_bytes(
        _canonical_json({"messages": messages or [], "tools": tools or []}).encode(
            "utf-8"
        )
    )


def _text_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        parts = []
        for block in value:
            if isinstance(block, Mapping) and isinstance(block.get("text"), str):
                parts.append(str(block["text"]))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(value)


def normalize_text_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Adapt OpenAI text-content blocks to a text-only Qwen chat template."""

    normalized = []
    for message in messages:
        current = dict(message)
        current["content"] = _text_content(message.get("content"))
        normalized.append(current)
    return normalized


def _normalize_arguments(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return {"value": value}
        return dict(decoded) if isinstance(decoded, Mapping) else {"value": decoded}
    return {"value": value}


def _normalize_call(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    name = str(value.get("name") or "").strip()
    if not name:
        return None
    return {"name": name, "arguments": _normalize_arguments(value.get("arguments"))}


def _balanced_json(text: str, start: int = 0) -> dict[str, Any] | None:
    object_start = text.find("{", start)
    if object_start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(object_start, len(text)):
        character = text[index]
        if escaped:
            escaped = False
            continue
        if character == "\\":
            escaped = True
            continue
        if character == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                try:
                    decoded = json.loads(text[object_start : index + 1])
                except json.JSONDecodeError:
                    return None
                return dict(decoded) if isinstance(decoded, Mapping) else None
    return None


def parse_main_tool_call(text: str) -> dict[str, Any] | None:
    marker = TOOL_CALL_RE.search(text or "")
    if marker is None:
        return None
    return _normalize_call(_balanced_json(text, marker.end()))


def parse_probe_tool_call(text: str) -> dict[str, Any] | None:
    return _normalize_call(_balanced_json('{"name": "' + (text or "")))


def exact_call_match(left: Any, right: Any) -> bool:
    normalized_left = _normalize_call(left)
    normalized_right = _normalize_call(right)
    return (
        normalized_left is not None
        and normalized_right is not None
        and normalized_left["name"] == normalized_right["name"]
        and _canonical_json(normalized_left["arguments"])
        == _canonical_json(normalized_right["arguments"])
    )


def _common_prefix_length(left: Sequence[int], right: Sequence[int]) -> int:
    length = 0
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        length += 1
    return length


def _find_subsequence(values: Sequence[int], needle: Sequence[int]) -> int | None:
    if not needle:
        return 0
    for index in range(len(values) - len(needle) + 1):
        if list(values[index : index + len(needle)]) == list(needle):
            return index
    return None


def span_min_probability(
    selected_logprobs: Sequence[float | None],
    *,
    max_positions: int = 21,
    skip_first: bool = True,
) -> float | None:
    values = list(selected_logprobs[:max_positions])
    if skip_first:
        values = values[1:]
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    return math.exp(min(finite)) if finite else None


def build_probe_prompt(main_prompt: str, observed_prefix: str) -> str:
    """Mirror SPORK's Qwen3 observed-prefix fork construction."""

    if "<think" in observed_prefix and "</think>" not in observed_prefix:
        return main_prompt + observed_prefix + "\n" + THINK_END_PREFIX
    return main_prompt + "<think>\n\n" + THINK_END_PREFIX


def _call_to_json(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    return _normalize_call(
        {
            "name": getattr(value, "name", None),
            "arguments": getattr(value, "arguments", None),
        }
    )


def load_actor_cases(
    tapes: Sequence[Path], *, actor_model: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load and canonical-hash deduplicate recorded Actor action requests."""

    cases_by_hash: dict[str, dict[str, Any]] = {}
    tape_metadata = []
    for tape_path in tapes:
        tape_hash = _sha256_file(tape_path)
        tape_metadata.append(
            {
                "name": tape_path.name,
                "sha256": tape_hash,
            }
        )
        parsed_by_sequence = {
            item.sequence: item
            for item in parse_tape(tape_path, "tagged_json")
            if item.model == actor_model and item.calls
        }
        raw = json.loads(tape_path.read_text(encoding="utf-8"))
        for exchange in raw.get("exchanges", []):
            sequence = int(exchange.get("sequence") or 0)
            parsed = parsed_by_sequence.get(sequence)
            if parsed is None:
                continue
            body = ((exchange.get("request") or {}).get("descriptor") or {}).get(
                "body"
            ) or {}
            if str(body.get("model") or "") != actor_model:
                continue
            messages = body.get("messages") or []
            tools = body.get("tools") or []
            request_hash = _request_hash(messages, tools)
            references = [
                value
                for value in (_call_to_json(call) for call in parsed.calls)
                if value is not None
            ]
            existing = cases_by_hash.get(request_hash)
            source = {
                "tape": tape_path.name,
                "tape_sha256": tape_hash,
                "sequence": sequence,
            }
            if existing is None:
                cases_by_hash[request_hash] = {
                    "request_hash": request_hash,
                    "messages": messages,
                    "tools": tools,
                    "reference_calls": references,
                    "sources": [source],
                }
            else:
                existing["sources"].append(source)
                for reference in references:
                    if not any(
                        exact_call_match(reference, current)
                        for current in existing["reference_calls"]
                    ):
                        existing["reference_calls"].append(reference)
    return list(cases_by_hash.values()), tape_metadata


def load_case_manifest(
    path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load model-output-free action cases with integrity-checked requests."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("case manifest must be a JSON object")
    if raw.get("format") != "self-speculation-action-case-manifest":
        raise ValueError("unsupported case manifest format")
    if raw.get("version") != 1:
        raise ValueError("unsupported case manifest version")
    raw_cases = raw.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("case manifest must contain at least one case")

    cases = []
    seen_hashes: set[str] = set()
    for index, value in enumerate(raw_cases):
        if not isinstance(value, Mapping):
            raise ValueError(f"case manifest entry {index} must be an object")
        messages = value.get("messages")
        tools = value.get("tools")
        if not isinstance(messages, list) or not isinstance(tools, list):
            raise ValueError(f"case manifest entry {index} has invalid messages/tools")
        expected_hash = str(value.get("request_hash") or "")
        actual_hash = _request_hash(messages, tools)
        if expected_hash != actual_hash:
            raise ValueError(f"case manifest entry {index} request hash mismatch")
        if actual_hash in seen_hashes:
            raise ValueError(f"case manifest entry {index} duplicates a request")
        seen_hashes.add(actual_hash)
        raw_references = value.get("reference_calls") or []
        if not isinstance(raw_references, list):
            raise ValueError(f"case manifest entry {index} has invalid references")
        references = [
            normalized
            for reference in raw_references
            if (normalized := _normalize_call(reference)) is not None
        ]
        sources = value.get("sources")
        if not isinstance(sources, list) or not all(
            isinstance(source, Mapping) for source in sources
        ):
            raise ValueError(f"case manifest entry {index} has invalid sources")
        case = {
            "request_hash": actual_hash,
            "messages": messages,
            "tools": tools,
            "reference_calls": references,
            "sources": sources,
        }
        case_id = value.get("case_id")
        if isinstance(case_id, str) and case_id:
            case["case_id"] = case_id
        cases.append(case)

    return cases, [
        {
            "name": path.name,
            "sha256": _sha256_file(path),
            "format": str(raw["format"]),
        }
    ]


def _sentence_boundary(text: str) -> bool:
    return bool(text) and (
        text.endswith("\n") or text.endswith(". ") or text.endswith(".\n")
    )


def next_probe_index(
    prefixes: Sequence[str],
    token_times_ms: Sequence[float],
    *,
    last_probe_index: int,
    prior_probe_end_ms: float,
    step: int = 50,
    sentence_slack: int = 30,
    snap_to_sentence: bool = True,
) -> int | None:
    """Counterfactually advance SPORK's retry loop on a recorded main stream."""

    if step <= 0:
        raise ValueError("step must be positive")
    target = last_probe_index + step
    token_count = min(len(prefixes), len(token_times_ms))
    if target > token_count:
        return None
    available = bisect.bisect_right(token_times_ms[:token_count], prior_probe_end_ms)
    candidate = max(target, available)
    if candidate > token_count:
        return None
    if not snap_to_sentence:
        return candidate
    forced = target + sentence_slack
    if candidate >= forced or _sentence_boundary(prefixes[candidate - 1]):
        return candidate
    for index in range(candidate + 1, min(forced, token_count) + 1):
        if index >= forced or _sentence_boundary(prefixes[index - 1]):
            return index
    return None


def _extract_stream_piece(event: Mapping[str, Any]) -> tuple[str, list[int]]:
    content = str(event.get("content") or "")
    tokens = [int(token) for token in event.get("tokens") or []]
    if content or tokens:
        return content, tokens
    choices = event.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
        choice = choices[0]
        return str(choice.get("text") or ""), [
            int(token) for token in choice.get("tokens") or []
        ]
    return "", []


class LlamaServerClient:
    def __init__(self, base_url: str, *, timeout_s: float = 600.0) -> None:
        try:
            import httpx
        except ImportError as error:  # pragma: no cover - exercised by CLI users
            raise RuntimeError("record mode requires the 'http' optional dependency") from error
        self._httpx = httpx
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def health(self) -> dict[str, Any]:
        response = self._httpx.get(
            f"{self.base_url}/health",
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        return dict(response.json())

    def stream_main(
        self,
        prompt: str,
        *,
        max_tokens: int,
        seed: int,
    ) -> dict[str, Any]:
        payload = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": 0.0,
            "seed": seed,
            "stop": ["</tool_call>"],
            "include_stop_str_in_output": True,
            "stream": True,
            "cache_prompt": True,
            "return_tokens": True,
        }
        started = time.perf_counter()
        text_parts: list[str] = []
        token_ids: list[int] = []
        token_times_ms: list[float] = []
        final_event: dict[str, Any] = {}
        with self._httpx.stream(
            "POST",
            f"{self.base_url}/completion",
            json=payload,
            timeout=self.timeout_s,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                event = json.loads(data)
                if not isinstance(event, Mapping):
                    continue
                final_event = dict(event)
                piece, tokens = _extract_stream_piece(event)
                text_parts.append(piece)
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                token_ids.extend(tokens)
                token_times_ms.extend([elapsed_ms] * len(tokens))
        return {
            "text": "".join(text_parts),
            "token_ids": token_ids,
            "token_times_ms": token_times_ms,
            "wall_ms": (time.perf_counter() - started) * 1000.0,
            "timings": final_event.get("timings") or {},
            "stop_type": final_event.get("stop_type"),
            "truncated": bool(final_event.get("truncated", False)),
        }

    def complete_probe(
        self,
        prompt: str,
        *,
        max_tokens: int,
        seed: int,
    ) -> dict[str, Any]:
        payload = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": 0.0,
            "seed": seed,
            "stop": ["</tool_call>"],
            "include_stop_str_in_output": True,
            "stream": False,
            "cache_prompt": True,
            "return_tokens": True,
            "n_probs": 5,
            "post_sampling_probs": False,
        }
        started = time.perf_counter()
        response = self._httpx.post(
            f"{self.base_url}/completion",
            json=payload,
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        result = dict(response.json())
        wall_ms = (time.perf_counter() - started) * 1000.0
        selected_logprobs = []
        for probability in result.get("completion_probabilities") or []:
            value = probability.get("logprob") if isinstance(probability, Mapping) else None
            selected_logprobs.append(
                float(value) if isinstance(value, (int, float)) else None
            )
        tokens = [int(token) for token in result.get("tokens") or []]
        return {
            "text": str(result.get("content") or ""),
            "token_ids": tokens,
            "token_count": int(result.get("tokens_predicted") or len(tokens)),
            "selected_logprobs": selected_logprobs,
            "wall_ms": wall_ms,
            "timings": result.get("timings") or {},
            "tokens_cached": result.get("tokens_cached"),
            "truncated": bool(result.get("truncated", False)),
        }


def _prefixes_from_tokens(tokenizer: Any, token_ids: Sequence[int]) -> list[str]:
    return [
        tokenizer.decode(token_ids[:index], skip_special_tokens=False)
        for index in range(1, len(token_ids) + 1)
    ]


def _repair_main_tokens(
    tokenizer: Any,
    main: dict[str, Any],
) -> None:
    token_ids = list(main.get("token_ids") or [])
    token_times = list(main.get("token_times_ms") or [])
    if not token_ids:
        token_ids = [
            int(token)
            for token in tokenizer.encode(
                str(main.get("text") or ""), add_special_tokens=False
            )
        ]
    if len(token_times) != len(token_ids):
        wall_ms = float(main.get("wall_ms") or 0.0)
        predicted_ms = float((main.get("timings") or {}).get("predicted_ms") or wall_ms)
        start_ms = max(0.0, wall_ms - predicted_ms)
        step_ms = predicted_ms / max(1, len(token_ids))
        token_times = [start_ms + step_ms * (index + 1) for index in range(len(token_ids))]
        main["token_timing_fallback"] = "linear_from_server_timings"
    main["token_ids"] = token_ids
    main["token_times_ms"] = token_times


def _probe_confident(probe: Mapping[str, Any], threshold: float) -> bool:
    confidence = probe.get("confidence")
    return (
        isinstance(confidence, (int, float))
        and math.isfinite(confidence)
        and confidence >= threshold
    )


def _probe_committable(probe: Mapping[str, Any], threshold: float) -> bool:
    return (
        _normalize_call(probe.get("call")) is not None
        and _probe_confident(probe, threshold)
    )


def record_case(
    client: LlamaServerClient,
    tokenizer: Any,
    case: Mapping[str, Any],
    *,
    main_max_tokens: int,
    probe_max_tokens: int,
    max_retries: int,
    retry_token_step: int,
    min_tokens_first_probe: int,
    seed: int,
    stop_after_confident_probe: bool = False,
    confidence_threshold: float = 0.90,
) -> dict[str, Any]:
    main_prompt = tokenizer.apply_chat_template(
        normalize_text_messages(case["messages"]),
        tools=case["tools"],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    main = client.stream_main(main_prompt, max_tokens=main_max_tokens, seed=seed)
    _repair_main_tokens(tokenizer, main)
    main["call"] = parse_main_tool_call(main["text"])
    prefixes = _prefixes_from_tokens(tokenizer, main["token_ids"])

    def record_probe(snapshot_index: int, retry_index: int) -> dict[str, Any]:
        observed_prefix = prefixes[snapshot_index - 1]
        snapshot_ms = float(main["token_times_ms"][snapshot_index - 1])
        probe = client.complete_probe(
            build_probe_prompt(main_prompt, observed_prefix),
            max_tokens=probe_max_tokens,
            seed=seed,
        )
        probe_end_ms = snapshot_ms + float(probe["wall_ms"])
        return {
            "retry_index": retry_index,
            "snapshot_token": snapshot_index,
            "snapshot_ms": snapshot_ms,
            "observed_prefix": observed_prefix,
            "text": probe["text"],
            "call": parse_probe_tool_call(probe["text"]),
            "confidence": span_min_probability(probe["selected_logprobs"]),
            "wall_ms": probe["wall_ms"],
            "token_ids": probe["token_ids"],
            "token_count": probe["token_count"],
            "selected_logprobs": probe["selected_logprobs"],
            "timings": probe["timings"],
            "tokens_cached": probe["tokens_cached"],
            "truncated": probe["truncated"],
            "optimistic_runway_ms": max(
                0.0,
                float(main["wall_ms"]) - probe_end_ms,
            ),
        }

    attempts = []
    snapshot_index = (
        min_tokens_first_probe
        if min_tokens_first_probe <= len(prefixes)
        else None
    )
    prior_probe_end_ms = 0.0
    for retry_index in range(max_retries):
        if snapshot_index is None or snapshot_index > len(prefixes):
            break
        attempt = record_probe(snapshot_index, retry_index)
        attempts.append(attempt)
        prior_probe_end_ms = float(attempt["snapshot_ms"]) + float(attempt["wall_ms"])
        if stop_after_confident_probe and _probe_committable(
            attempt,
            confidence_threshold,
        ):
            break
        snapshot_index = next_probe_index(
            prefixes,
            main["token_times_ms"],
            last_probe_index=snapshot_index,
            prior_probe_end_ms=prior_probe_end_ms,
            step=retry_token_step,
        )
    # Record the control last so it cannot evict the main's long prefix before
    # the delayed policy is measured on a single-slot prefix cache.
    d1_probe = (
        record_probe(1, -1)
        if prefixes and min_tokens_first_probe > 1
        else None
    )

    result = {
        "request_hash": case["request_hash"],
        "sources": case["sources"],
        "reference_calls": case["reference_calls"],
        "main": main,
        "d1_probe": d1_probe,
        "probes": attempts,
    }
    if isinstance(case.get("case_id"), str):
        result["case_id"] = case["case_id"]
    return result


def _select_d1(turn: Mapping[str, Any]) -> dict[str, Any]:
    probes = list(turn.get("probes") or [])
    first = turn.get("d1_probe")
    if not isinstance(first, Mapping):
        first = probes[0] if probes else None
    committed = first if first and _normalize_call(first.get("call")) else None
    main_call = (turn.get("main") or {}).get("call")
    return {
        "attempts": int(first is not None),
        "probe_tokens": int(first.get("token_count") or 0) if first else 0,
        "committed": committed,
        "dispatched": committed is not None,
        "exact": bool(committed and exact_call_match(committed.get("call"), main_call)),
    }


def _select_d2(turn: Mapping[str, Any], threshold: float) -> dict[str, Any]:
    probes = list(turn.get("probes") or [])
    committed = None
    attempts = 0
    probe_tokens = 0
    for probe in probes:
        attempts += 1
        probe_tokens += int(probe.get("token_count") or 0)
        if _probe_committable(probe, threshold):
            committed = probe
            break
    main_call = (turn.get("main") or {}).get("call")
    return {
        "attempts": attempts,
        "probe_tokens": probe_tokens,
        "committed": committed,
        "dispatched": committed is not None,
        "exact": bool(committed and exact_call_match(committed.get("call"), main_call)),
    }


def _phase1_early_abort_tokens(
    turn: Mapping[str, Any],
    *,
    threshold: float,
    span_tokens: int = 20,
    first_probe_full: bool = False,
) -> int:
    decision = _select_d2(turn, threshold)
    probes = list(turn.get("probes") or [])[: int(decision["attempts"])]
    total = 0
    for index, probe in enumerate(probes):
        full_tokens = int(probe.get("token_count") or 0)
        total += (
            full_tokens
            if (first_probe_full and index == 0)
            or _probe_confident(probe, threshold)
            else min(span_tokens, full_tokens)
        )
    return total


def analyze_recording(recording: Mapping[str, Any], *, threshold: float = 0.90) -> dict[str, Any]:
    turns = list(recording.get("turns") or [])
    config = recording.get("config") or {}
    eligible = [turn for turn in turns if _normalize_call((turn.get("main") or {}).get("call"))]
    d1_decisions = [_select_d1(turn) for turn in eligible]
    d2_decisions = [_select_d2(turn, threshold) for turn in eligible]
    d1_hits = sum(bool(decision["exact"]) for decision in d1_decisions)
    d2_hits = sum(bool(decision["exact"]) for decision in d2_decisions)
    d1_dispatches = sum(bool(decision["dispatched"]) for decision in d1_decisions)
    d2_dispatches = sum(bool(decision["dispatched"]) for decision in d2_decisions)
    recovered = [
        (turn, d2)
        for turn, d1, d2 in zip(eligible, d1_decisions, d2_decisions)
        if not d1["exact"] and d2["exact"]
    ]
    lost = sum(
        bool(d1["exact"] and not d2["exact"])
        for d1, d2 in zip(d1_decisions, d2_decisions)
    )
    oracle_hits = sum(
        any(
            exact_call_match(probe.get("call"), (turn.get("main") or {}).get("call"))
            for probe in turn.get("probes") or []
        )
        for turn in eligible
    )
    portability_hits = sum(
        any(
            exact_call_match((turn.get("main") or {}).get("call"), reference)
            for reference in turn.get("reference_calls") or []
        )
        for turn in eligible
    )
    d1_tokens = sum(int(decision["probe_tokens"]) for decision in d1_decisions)
    d2_tokens = sum(int(decision["probe_tokens"]) for decision in d2_decisions)
    configured_phase1_span = config.get("phase1_span_tokens")
    phase1_span = (
        int(configured_phase1_span)
        if isinstance(configured_phase1_span, int) and configured_phase1_span > 0
        else 20
    )
    phase1_early_abort_tokens = sum(
        _phase1_early_abort_tokens(
            turn,
            threshold=threshold,
            span_tokens=phase1_span,
            first_probe_full=bool(config.get("phase1_first_probe_full")),
        )
        for turn in eligible
    )
    mean_attempts = (
        sum(int(decision["attempts"]) for decision in d2_decisions) / len(eligible)
        if eligible
        else 0.0
    )
    token_ratio = d2_tokens / d1_tokens if d1_tokens else None
    efficiency_tokens = (
        phase1_early_abort_tokens
        if isinstance(configured_phase1_span, int) and configured_phase1_span > 0
        else d2_tokens
    )
    efficiency_token_ratio = efficiency_tokens / d1_tokens if d1_tokens else None
    d1_precision = d1_hits / d1_dispatches if d1_dispatches else 0.0
    d2_precision = d2_hits / d2_dispatches if d2_dispatches else 0.0
    recovered_runways = [
        float((decision.get("committed") or {}).get("optimistic_runway_ms") or 0.0)
        for _, decision in recovered
    ]
    max_mean_attempts = float(config.get("gate_max_mean_probe_attempts") or 2.0)
    max_token_ratio = float(config.get("gate_max_probe_token_ratio") or 1.75)
    gates = {
        "validity": len(eligible) >= 6,
        "recall": d2_hits >= d1_hits and lost == 0 and len(recovered) >= 1,
        "precision": d2_precision >= d1_precision,
        "probe_efficiency": mean_attempts <= max_mean_attempts
        and efficiency_token_ratio is not None
        and efficiency_token_ratio <= max_token_ratio,
        "usable_recovery": bool(recovered_runways)
        and all(runway >= 25.0 for runway in recovered_runways),
    }
    return {
        "threshold": threshold,
        "recorded_turns": len(turns),
        "eligible_main_tool_turns": len(eligible),
        "local_main_matches_recorded_deepseek": portability_hits,
        "d1": {
            "dispatches": d1_dispatches,
            "exact_hits": d1_hits,
            "precision": d1_precision,
            "probe_tokens": d1_tokens,
        },
        "d1_d2": {
            "dispatches": d2_dispatches,
            "exact_hits": d2_hits,
            "precision": d2_precision,
            "probe_attempts": sum(int(value["attempts"]) for value in d2_decisions),
            "mean_probe_attempts": mean_attempts,
            "probe_tokens": d2_tokens,
            "probe_token_ratio_vs_d1": token_ratio,
            "efficiency_accounting": {
                "policy": (
                    (
                        f"d1_full_plus_d2_phase1_{phase1_span}_token_early_abort"
                        if config.get("phase1_first_probe_full")
                        else f"phase1_{phase1_span}_token_early_abort"
                    )
                    if isinstance(configured_phase1_span, int)
                    and configured_phase1_span > 0
                    else "full_probe"
                ),
                "probe_tokens": efficiency_tokens,
                "probe_token_ratio_vs_d1": efficiency_token_ratio,
                "gate_max_mean_probe_attempts": max_mean_attempts,
                "gate_max_probe_token_ratio": max_token_ratio,
            },
            "phase1_20_token_early_abort": {
                "probe_tokens": phase1_early_abort_tokens,
                "probe_token_ratio_vs_d1": (
                    phase1_early_abort_tokens / d1_tokens if d1_tokens else None
                ),
            },
            "recovered_hits": len(recovered),
            "lost_d1_hits": lost,
            "recovered_optimistic_runway_ms": recovered_runways,
        },
        "d2_oracle_exact_hits": oracle_hits,
        "gates": gates,
        "product_gate_passed": all(gates.values()),
    }


def _simulate_serial_bundle(
    actual: Sequence[int],
    candidates: Sequence[Sequence[int]],
    *,
    limit: int,
) -> dict[str, int]:
    """Simulate the existing serial target-verifier fallback contract."""

    generated = 0
    proposals = 0
    proposed_tokens = 0
    accepted_tokens = 0
    for full_candidate in candidates:
        candidate = tuple(int(token) for token in full_candidate[:limit])
        if generated > len(candidate) or tuple(actual[:generated]) != candidate[:generated]:
            continue
        proposal = candidate[generated:]
        if not proposal or generated >= len(actual):
            continue
        accepted = _common_prefix_length(proposal, actual[generated:])
        proposals += 1
        proposed_tokens += len(proposal)
        accepted_tokens += accepted
        generated += accepted
        if generated < len(actual):
            generated += 1
    return {
        "target_steps": proposals + max(0, len(actual) - generated),
        "proposals": proposals,
        "proposed_tokens": proposed_tokens,
        "accepted_tokens": accepted_tokens,
    }


def analyze_d3_recording(
    recording: Mapping[str, Any],
    tokenizer: Any,
    *,
    threshold: float = 0.90,
) -> dict[str, Any]:
    """Measure boundary-draft reuse from recorded D1/D2 probe continuations."""

    marker = [
        int(token)
        for token in tokenizer.encode(
            '<tool_call>\n{"name": "',
            add_special_tokens=False,
        )
    ]
    rows = []
    for turn in recording.get("turns") or []:
        main = turn.get("main") or {}
        if _normalize_call(main.get("call")) is None:
            continue
        main_tokens = [int(token) for token in main.get("token_ids") or []]
        token_times = [float(value) for value in main.get("token_times_ms") or []]
        marker_index = _find_subsequence(main_tokens, marker)
        if marker_index is None or marker_index >= len(token_times):
            continue
        continuation_start = marker_index + len(marker)
        actual = main_tokens[continuation_start:]
        boundary_ms = token_times[marker_index]

        def boundary_attempt(probe: Mapping[str, Any]) -> dict[str, Any]:
            end_ms = float(probe.get("snapshot_ms") or 0.0) + float(
                probe.get("wall_ms") or 0.0
            )
            call = _normalize_call(probe.get("call"))
            tokens = [int(token) for token in probe.get("token_ids") or []]
            return {
                "probe": probe,
                "parseable": call is not None,
                "available": call is not None and end_ms <= boundary_ms,
                "tokens": tokens,
                "accepted": _common_prefix_length(tokens, actual),
            }

        attempts = []
        for probe in turn.get("probes") or []:
            attempts.append(boundary_attempt(probe))
        raw_d1_probe = turn.get("d1_probe")
        if not isinstance(raw_d1_probe, Mapping):
            raw_d1_probe = (turn.get("probes") or [None])[0]
        d1_view = (
            boundary_attempt(raw_d1_probe)
            if isinstance(raw_d1_probe, Mapping)
            else None
        )
        d1 = d1_view if d1_view and d1_view["available"] else None
        d2_decision = _select_d2(turn, threshold)
        considered = attempts[: int(d2_decision["attempts"])]
        d2_available = [attempt for attempt in considered if attempt["available"]]
        d2 = d2_available[-1] if d2_available else None
        committed_probe = d2_decision.get("committed")
        committed_view = (
            boundary_attempt(committed_probe)
            if isinstance(committed_probe, Mapping)
            else None
        )
        bundle_candidates = []
        for view in (committed_view, d1):
            if not view or not view["available"]:
                continue
            candidate = tuple(view["tokens"])
            if candidate and candidate not in bundle_candidates:
                bundle_candidates.append(candidate)
        d1_candidates = [tuple(d1["tokens"])] if d1 else []
        d1_serial = _simulate_serial_bundle(actual, d1_candidates, limit=28)
        bundle_serial = _simulate_serial_bundle(actual, bundle_candidates, limit=28)
        all_available = [attempt for attempt in attempts if attempt["available"]]
        latest = all_available[-1] if all_available else None
        rows.append(
            {
                "continuation_tokens": len(actual),
                "d1_accepted": int(d1["accepted"]) if d1 else 0,
                "d2_accepted": int(d2["accepted"]) if d2 else 0,
                "latest_accepted": int(latest["accepted"]) if latest else 0,
                "oracle_accepted": max(
                    (int(attempt["accepted"]) for attempt in all_available),
                    default=0,
                ),
                "all_probe_tokens": sum(
                    int(attempt["probe"].get("token_count") or 0)
                    for attempt in attempts
                ),
                "d1_serial": d1_serial,
                "bundle_serial": bundle_serial,
            }
        )
    d2_summary = analyze_recording(recording, threshold=threshold)
    d1_probe_tokens = int(d2_summary["d1"]["probe_tokens"])
    d2_probe_tokens = int(d2_summary["d1_d2"]["probe_tokens"])
    d1_accepted = sum(row["d1_accepted"] for row in rows)
    d2_accepted = sum(row["d2_accepted"] for row in rows)
    marginal_accepted = d2_accepted - d1_accepted
    marginal_probe_tokens = d2_probe_tokens - d1_probe_tokens
    phase1_probe_tokens = int(
        d2_summary["d1_d2"]["phase1_20_token_early_abort"]["probe_tokens"]
    )
    d1_serial = {
        key: sum(row["d1_serial"][key] for row in rows)
        for key in ("target_steps", "proposals", "proposed_tokens", "accepted_tokens")
    }
    bundle_serial = {
        key: sum(row["bundle_serial"][key] for row in rows)
        for key in ("target_steps", "proposals", "proposed_tokens", "accepted_tokens")
    }
    bundle_gates = {
        "accepted_tokens_no_regression": (
            bundle_serial["accepted_tokens"] >= d1_serial["accepted_tokens"]
        ),
        "target_steps_no_regression": (
            bundle_serial["target_steps"] <= d1_serial["target_steps"]
        ),
    }
    return {
        "tool_turns_with_token_boundary": len(rows),
        "tool_continuation_tokens": sum(row["continuation_tokens"] for row in rows),
        "d1": {
            "accepted_target_tokens": d1_accepted,
            "probe_tokens": d1_probe_tokens,
        },
        "d1_d2": {
            "accepted_target_tokens": d2_accepted,
            "probe_tokens": d2_probe_tokens,
            "marginal_accepted_target_tokens": marginal_accepted,
            "marginal_probe_tokens": marginal_probe_tokens,
            "marginal_probe_tokens_per_accepted_target_token": (
                marginal_probe_tokens / marginal_accepted
                if marginal_accepted > 0
                else None
            ),
            "phase1_20_token_early_abort": {
                "probe_tokens": phase1_probe_tokens,
                "marginal_probe_tokens": phase1_probe_tokens - d1_probe_tokens,
                "marginal_probe_tokens_per_accepted_target_token": (
                    (phase1_probe_tokens - d1_probe_tokens) / marginal_accepted
                    if marginal_accepted > 0
                    else None
                ),
            },
        },
        "continue_all_reprobes_latest": {
            "accepted_target_tokens": sum(row["latest_accepted"] for row in rows),
            "probe_tokens": sum(row["all_probe_tokens"] for row in rows),
        },
        "available_parseable_oracle": {
            "accepted_target_tokens": sum(row["oracle_accepted"] for row in rows),
        },
        "bounded_d2_then_d1_bundle_k28": {
            "d1": d1_serial,
            "d1_d2": bundle_serial,
            "gates": bundle_gates,
            "passed": all(bundle_gates.values()),
        },
    }


def _attach_d3_analysis(
    result: dict[str, Any],
    recording: Mapping[str, Any],
    tokenizer: Any,
    *,
    threshold: float,
) -> None:
    d3 = analyze_d3_recording(recording, tokenizer, threshold=threshold)
    result["d3_boundary_reuse"] = d3
    if not (recording.get("config") or {}).get("require_d3_bundle_gate"):
        return
    passed = bool(d3["bounded_d2_then_d1_bundle_k28"]["passed"])
    result["gates"]["d3_bundle_no_regression"] = passed
    result["product_gate_passed"] = all(bool(value) for value in result["gates"].values())


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _record(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, revision=args.revision)
    if args.case_manifest is not None:
        cases, tapes = load_case_manifest(args.case_manifest)
    else:
        cases, tapes = load_actor_cases(args.tape, actor_model=args.actor_model)
    if args.limit is not None:
        cases = cases[: args.limit]
    client = LlamaServerClient(args.server_url, timeout_s=args.timeout_s)
    client.health()
    record_stop_threshold = (
        args.record_stop_confidence_threshold
        if args.record_stop_confidence_threshold is not None
        else args.confidence_threshold
    )
    config = {
        "actor_model": args.actor_model,
        "tokenizer": args.tokenizer,
        "tokenizer_revision": args.revision,
        "model": args.model,
        "model_revision": args.model_revision,
        "llama_cpp_build": args.llama_cpp_build,
        "main_max_tokens": args.main_max_tokens,
        "probe_max_tokens": args.probe_max_tokens,
        "max_retries": args.max_retries,
        "retry_token_step": args.retry_token_step,
        "min_tokens_first_probe": args.min_tokens_first_probe,
        "phase1_span_tokens": args.phase1_span_tokens,
        "confidence_threshold": args.confidence_threshold,
        "seed": args.seed,
    }
    if args.stop_after_confident_probe:
        config["stop_after_confident_probe"] = True
        config["record_stop_confidence_threshold"] = record_stop_threshold
    if args.phase1_first_probe_full:
        config["phase1_first_probe_full"] = True
    if args.gate_max_mean_probe_attempts != 2.0:
        config["gate_max_mean_probe_attempts"] = args.gate_max_mean_probe_attempts
    if args.gate_max_probe_token_ratio != 1.75:
        config["gate_max_probe_token_ratio"] = args.gate_max_probe_token_ratio
    if args.require_d3_bundle_gate:
        config["require_d3_bundle_gate"] = True
    existing_by_hash: dict[str, dict[str, Any]] = {}
    if args.resume and args.output.exists():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        if existing.get("config") != config:
            raise ValueError("resume recording configuration does not match")
        existing_by_hash = {
            str(turn.get("request_hash")): turn for turn in existing.get("turns") or []
        }
    recording: dict[str, Any] = {
        "format": "self-speculation-d2-think-replay",
        "version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "config": config,
        "tapes": tapes,
        "turns": [],
    }
    for index, case in enumerate(cases, start=1):
        request_hash = str(case["request_hash"])
        turn = existing_by_hash.get(request_hash)
        if turn is None:
            print(f"recording {index}/{len(cases)} {request_hash[:12]}", flush=True)
            try:
                turn = record_case(
                    client,
                    tokenizer,
                    case,
                    main_max_tokens=args.main_max_tokens,
                    probe_max_tokens=args.probe_max_tokens,
                    max_retries=args.max_retries,
                    retry_token_step=args.retry_token_step,
                    min_tokens_first_probe=args.min_tokens_first_probe,
                    stop_after_confident_probe=args.stop_after_confident_probe,
                    confidence_threshold=record_stop_threshold,
                    seed=args.seed,
                )
            except Exception as error:  # keep long-running recordings resumable
                turn = {
                    "request_hash": request_hash,
                    "sources": case["sources"],
                    "reference_calls": case["reference_calls"],
                    "error": f"{type(error).__name__}: {error}",
                    "main": {},
                    "probes": [],
                }
                if isinstance(case.get("case_id"), str):
                    turn["case_id"] = case["case_id"]
        recording["turns"].append(turn)
        recording["analysis"] = analyze_recording(
            recording, threshold=args.confidence_threshold
        )
        _write_json_atomic(args.output, recording)
    recording["sensitivity"] = {
        str(threshold): analyze_recording(recording, threshold=threshold)
        for threshold in (0.85, 0.90, 0.95)
    }
    if args.require_d3_bundle_gate:
        _attach_d3_analysis(
            recording["analysis"],
            recording,
            tokenizer,
            threshold=args.confidence_threshold,
        )
    _write_json_atomic(args.output, recording)
    return recording


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    record = subparsers.add_parser("record")
    inputs = record.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--tape", type=Path, action="append")
    inputs.add_argument("--case-manifest", type=Path)
    record.add_argument("--output", type=Path, required=True)
    record.add_argument("--server-url", default="http://127.0.0.1:18080")
    record.add_argument("--actor-model", default="deepseek-v4-pro")
    record.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    record.add_argument("--revision", default=DEFAULT_TOKENIZER_REVISION)
    record.add_argument("--model", default="Qwen/Qwen3-1.7B-GGUF:Q8_0")
    record.add_argument(
        "--model-revision",
        default="90862c4b9d2787eaed51d12237eafdfe7c5f6077",
    )
    record.add_argument("--llama-cpp-build", default="10615:f280b2698")
    record.add_argument("--main-max-tokens", type=int, default=3072)
    record.add_argument("--probe-max-tokens", type=int, default=100)
    record.add_argument("--max-retries", type=int, default=5)
    record.add_argument("--retry-token-step", type=int, default=50)
    record.add_argument("--min-tokens-first-probe", type=int, default=1)
    record.add_argument("--stop-after-confident-probe", action="store_true")
    record.add_argument("--record-stop-confidence-threshold", type=float)
    record.add_argument("--phase1-span-tokens", type=int)
    record.add_argument("--phase1-first-probe-full", action="store_true")
    record.add_argument("--confidence-threshold", type=float, default=0.90)
    record.add_argument("--gate-max-mean-probe-attempts", type=float, default=2.0)
    record.add_argument("--gate-max-probe-token-ratio", type=float, default=1.75)
    record.add_argument("--require-d3-bundle-gate", action="store_true")
    record.add_argument("--seed", type=int, default=42)
    record.add_argument("--timeout-s", type=float, default=600.0)
    record.add_argument("--limit", type=int)
    record.add_argument("--resume", action="store_true")

    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--recording", type=Path, required=True)
    analyze.add_argument("--confidence-threshold", type=float, default=0.90)
    analyze.add_argument("--d3", action="store_true")
    analyze.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    analyze.add_argument("--revision", default=DEFAULT_TOKENIZER_REVISION)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "record":
        if min(
            args.main_max_tokens,
            args.probe_max_tokens,
            args.max_retries,
            args.retry_token_step,
            args.min_tokens_first_probe,
        ) <= 0:
            raise SystemExit("token limits, retries, and retry step must be positive")
        if args.phase1_span_tokens is not None and args.phase1_span_tokens <= 0:
            raise SystemExit("--phase1-span-tokens must be positive when provided")
        if args.phase1_first_probe_full and args.phase1_span_tokens is None:
            raise SystemExit("--phase1-first-probe-full requires --phase1-span-tokens")
        if not 0.0 < args.confidence_threshold <= 1.0:
            raise SystemExit("--confidence-threshold must be in (0, 1]")
        if args.record_stop_confidence_threshold is not None and not (
            0.0 < args.record_stop_confidence_threshold <= 1.0
        ):
            raise SystemExit("--record-stop-confidence-threshold must be in (0, 1]")
        if (
            args.record_stop_confidence_threshold is not None
            and not args.stop_after_confident_probe
        ):
            raise SystemExit(
                "--record-stop-confidence-threshold requires "
                "--stop-after-confident-probe"
            )
        if min(
            args.gate_max_mean_probe_attempts,
            args.gate_max_probe_token_ratio,
        ) <= 0 or not all(
            math.isfinite(value)
            for value in (
                args.gate_max_mean_probe_attempts,
                args.gate_max_probe_token_ratio,
            )
        ):
            raise SystemExit("efficiency gate limits must be positive")
        recording = _record(args)
        result = {
            "output": str(args.output),
            "analysis": recording["analysis"],
            "sensitivity": recording["sensitivity"],
        }
    else:
        recording = json.loads(args.recording.read_text(encoding="utf-8"))
        result = analyze_recording(
            recording,
            threshold=args.confidence_threshold,
        )
        if args.d3:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                args.tokenizer,
                revision=args.revision,
            )
            _attach_d3_analysis(
                result,
                recording,
                tokenizer,
                threshold=args.confidence_threshold,
            )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
