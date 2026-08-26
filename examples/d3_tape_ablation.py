"""Measure fixed D3 draft lengths against strict recorded tool calls.

This is an offline verifier-work proxy. It does not claim wall-clock speedup:
the tape contains provider request timing, not target-model speculative rounds.
"""

from __future__ import annotations

import argparse
import base64
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from self_speculation import ToolCall, format_tool_call_draft


@dataclass(frozen=True, slots=True)
class ParsedExchange:
    sequence: int
    model: str
    context_key: str
    duration_ms: float
    calls: tuple[ToolCall, ...]


@dataclass(frozen=True, slots=True)
class DraftLengthResult:
    ordering: str
    max_draft_tokens: int
    opportunities: int
    candidate_count: int
    actor_tokens: int
    proposals: int
    proposed_tokens: int
    accepted_tokens: int
    rejected_tokens: int
    acceptance_rate: float
    target_steps: int
    target_steps_saved: int
    target_step_reduction: float


@dataclass(frozen=True, slots=True)
class DraftOpportunity:
    sequence: int
    actual_tokens: tuple[int, ...]
    candidate_tokens: tuple[tuple[int, ...], ...]


def _context_key(messages: Any) -> str:
    return json.dumps(
        messages or [],
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sse_events(chunks: Iterable[Mapping[str, Any]]) -> Iterable[Mapping[str, Any]]:
    buffered = ""
    for chunk in chunks:
        buffered += base64.b64decode(str(chunk.get("dataBase64") or "")).decode(
            "utf-8", errors="replace"
        )
        blocks = buffered.replace("\r\n", "\n").split("\n\n")
        buffered = blocks.pop()
        for block in blocks:
            data = "\n".join(
                line[5:].strip() for line in block.splitlines() if line.startswith("data:")
            )
            if not data or data == "[DONE]":
                continue
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue
            if isinstance(event, Mapping):
                yield event


def _decode_calls(chunks: Iterable[Mapping[str, Any]], format_name: str) -> tuple[ToolCall, ...]:
    fragments: dict[int, dict[str, str]] = defaultdict(lambda: {"name": "", "arguments": ""})
    for event in _sse_events(chunks):
        choices = event.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
            continue
        delta = choices[0].get("delta") or choices[0].get("message")
        if not isinstance(delta, Mapping):
            continue
        raw_calls = delta.get("tool_calls")
        if not isinstance(raw_calls, list):
            continue
        for raw_call in raw_calls:
            if not isinstance(raw_call, Mapping):
                continue
            index = int(raw_call.get("index") or 0)
            function = raw_call.get("function")
            if not isinstance(function, Mapping):
                continue
            fragments[index]["name"] += str(function.get("name") or "")
            fragments[index]["arguments"] += str(function.get("arguments") or "")

    calls = []
    for index, fragment in sorted(fragments.items()):
        if not fragment["name"].strip():
            continue
        try:
            arguments = json.loads(fragment["arguments"] or "{}")
        except json.JSONDecodeError:
            continue
        calls.append(
            ToolCall(
                name=fragment["name"],
                arguments=arguments,
                index=index,
                format=format_name,
            )
        )
    return tuple(calls)


def parse_tape(path: Path, format_name: str) -> tuple[ParsedExchange, ...]:
    tape = json.loads(path.read_text(encoding="utf-8"))
    parsed = []
    for exchange in tape.get("exchanges", []):
        response = exchange.get("response") or {}
        if response.get("completed") is not True:
            continue
        body = ((exchange.get("request") or {}).get("descriptor") or {}).get("body") or {}
        model = str(body.get("model") or "").strip()
        duration = response.get("endedAtMs")
        if not model or not isinstance(duration, (int, float)) or duration < 0:
            continue
        parsed.append(
            ParsedExchange(
                sequence=int(exchange.get("sequence") or 0),
                model=model,
                context_key=_context_key(body.get("messages")),
                duration_ms=float(duration),
                calls=_decode_calls(response.get("chunks") or (), format_name),
            )
        )
    return tuple(parsed)


def _common_prefix(left: Sequence[int], right: Sequence[int]) -> int:
    length = 0
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        length += 1
    return length


def _order_candidates(
    candidates: Sequence[Sequence[int]], *, limit: int, ordering: str
) -> tuple[Sequence[int], ...]:
    """Order an equal-confidence group without consulting its source label.

    Pairwise prefix consensus is the medoid objective for the verifier: under
    an uninformative (uniform) prior over the tied candidates, it maximizes the
    expected number of immediately accepted draft tokens.
    """

    if ordering == "completion":
        return tuple(candidates)
    if ordering != "prefix-consensus":
        raise ValueError(f"unsupported candidate ordering: {ordering}")

    truncated = tuple(tuple(candidate[:limit]) for candidate in candidates)
    consensus = tuple(
        sum(_common_prefix(candidate, peer) for peer in truncated)
        for candidate in truncated
    )
    return tuple(
        candidates[index]
        for index in sorted(
            range(len(candidates)),
            key=lambda index: (-consensus[index], index),
        )
    )


def _simulate(
    actual: Sequence[int], candidates: Sequence[Sequence[int]], limit: int
) -> tuple[int, int, int, int]:
    """Return target steps, proposal count, proposed tokens, and accepted tokens."""

    generated = 0
    proposals = 0
    proposed_tokens = 0
    accepted_tokens = 0
    for full_candidate in candidates:
        candidate = tuple(full_candidate[:limit])
        if generated > len(candidate) or tuple(actual[:generated]) != candidate[:generated]:
            continue
        proposal = candidate[generated:]
        if not proposal or generated >= len(actual):
            continue
        accepted = _common_prefix(proposal, actual[generated:])
        proposals += 1
        proposed_tokens += len(proposal)
        accepted_tokens += accepted
        generated += accepted
        if generated < len(actual):
            # One verifier round also samples the first target token after the accepted prefix.
            generated += 1
    target_steps = proposals + max(0, len(actual) - generated)
    return target_steps, proposals, proposed_tokens, accepted_tokens


def build_opportunities(
    exchanges: Sequence[ParsedExchange],
    *,
    actor_model: str,
    drafter_model: str,
    tokenizer: Any,
    drafter_width: int | None = None,
) -> tuple[DraftOpportunity, ...]:
    if drafter_width is not None and drafter_width <= 0:
        raise ValueError("drafter_width must be positive when provided")
    drafters: dict[str, list[ParsedExchange]] = defaultdict(list)
    for exchange in exchanges:
        if exchange.model == drafter_model:
            drafters[exchange.context_key].append(exchange)
    for values in drafters.values():
        values.sort(key=lambda item: (item.sequence, item.duration_ms))
        if drafter_width is not None:
            del values[drafter_width:]
        values.sort(key=lambda item: (item.duration_ms, item.sequence))

    opportunities = []
    for actor in exchanges:
        if actor.model != actor_model:
            continue
        candidate_calls = [
            call
            for draft in drafters.get(actor.context_key, ())
            for call in draft.calls
        ]
        distinct: list[tuple[int, ...]] = []
        seen: set[tuple[int, ...]] = set()
        for call in candidate_calls:
            tokens = tuple(
                int(token)
                for token in tokenizer.encode(
                    format_tool_call_draft((call,)), add_special_tokens=False
                )
            )
            if tokens not in seen:
                seen.add(tokens)
                distinct.append(tokens)
        for call in actor.calls:
            actual = tuple(
                int(token)
                for token in tokenizer.encode(
                    format_tool_call_draft((call,)), add_special_tokens=False
                )
            )
            opportunities.append(
                DraftOpportunity(
                    sequence=actor.sequence,
                    actual_tokens=actual,
                    candidate_tokens=tuple(distinct),
                )
            )
    return tuple(opportunities)


def analyze(
    exchanges: Sequence[ParsedExchange],
    *,
    actor_model: str,
    drafter_model: str,
    tokenizer: Any,
    limits: Sequence[int],
    orderings: Sequence[str] = ("completion",),
) -> tuple[DraftLengthResult, ...]:
    opportunities = build_opportunities(
        exchanges,
        actor_model=actor_model,
        drafter_model=drafter_model,
        tokenizer=tokenizer,
    )

    results = []
    for ordering in orderings:
        for limit in limits:
            actor_tokens = sum(
                len(opportunity.actual_tokens) for opportunity in opportunities
            )
            candidate_count = sum(
                len(opportunity.candidate_tokens)
                for opportunity in opportunities
            )
            proposals = proposed = accepted = target_steps = 0
            for opportunity in opportunities:
                ranked = _order_candidates(
                    opportunity.candidate_tokens,
                    limit=limit,
                    ordering=ordering,
                )
                steps, proposal_count, offered, matched = _simulate(
                    opportunity.actual_tokens,
                    ranked,
                    limit,
                )
                target_steps += steps
                proposals += proposal_count
                proposed += offered
                accepted += matched
            rejected = proposed - accepted
            saved = actor_tokens - target_steps
            results.append(
                DraftLengthResult(
                    ordering=ordering,
                    max_draft_tokens=limit,
                    opportunities=len(opportunities),
                    candidate_count=candidate_count,
                    actor_tokens=actor_tokens,
                    proposals=proposals,
                    proposed_tokens=proposed,
                    accepted_tokens=accepted,
                    rejected_tokens=rejected,
                    acceptance_rate=accepted / proposed if proposed else 0.0,
                    target_steps=target_steps,
                    target_steps_saved=saved,
                    target_step_reduction=saved / actor_tokens if actor_tokens else 0.0,
                )
            )
    return tuple(results)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tape", type=Path, required=True)
    parser.add_argument("--actor-model", required=True)
    parser.add_argument("--drafter-model", required=True)
    parser.add_argument("--tokenizer", default="deepseek-ai/DeepSeek-V3")
    parser.add_argument("--revision")
    parser.add_argument("--format", default="tagged_json")
    parser.add_argument("--limits", default="4,8,12,20,32")
    parser.add_argument("--orderings", default="completion,prefix-consensus")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, revision=args.revision)
    limits = tuple(int(item) for item in args.limits.split(",") if int(item) > 0)
    orderings = tuple(
        item.strip() for item in args.orderings.split(",") if item.strip()
    )
    results = analyze(
        parse_tape(args.tape, args.format),
        actor_model=args.actor_model,
        drafter_model=args.drafter_model,
        tokenizer=tokenizer,
        limits=limits,
        orderings=orderings,
    )
    print(
        json.dumps(
            {
                "tape": str(args.tape),
                "tokenizer": args.tokenizer,
                "tokenizer_revision": args.revision or tokenizer.init_kwargs.get("_commit_hash"),
                "format": args.format,
                "results": [asdict(result) for result in results],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
