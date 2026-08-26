"""Causal, suffix-decoding-inspired replay over prior tool-call bodies."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RECORDING_SHA256 = "da2ca1b8c5851c4ffceb0423602e636b507536b76b9418594fb3fc5273eef0e3"
TOKENIZER_REPOSITORY = "Qwen/Qwen3-1.7B"
TOKENIZER_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
MAX_HISTORY = 64
MAX_BODY_BYTES = 4_096
MAX_DRAFT = 28
MIN_TOKEN_PROBABILITY = 0.1
I30_TARGET_STEPS = 2_381


@dataclass(frozen=True, slots=True)
class Occurrence:
    tokens: tuple[int, ...]
    order: int


@dataclass(frozen=True, slots=True)
class Replay:
    target_steps: int
    proposals: int
    proposed: int
    accepted: int
    max_proposal: int


def qwen_body(call: Mapping[str, Any]) -> str:
    name = str(call.get("name") or "").strip()
    if not name:
        raise ValueError("call name must not be empty")
    arguments = call.get("arguments")
    if not isinstance(arguments, Mapping):
        arguments = {}
    return "\n" + json.dumps(
        {"name": name, "arguments": arguments},
        ensure_ascii=False,
        separators=(", ", ": "),
    ) + "\n"


def trie_proposal(
    history: Sequence[Occurrence], generated_prefix: Sequence[int]
) -> tuple[int, ...]:
    """Choose a frequency/recency path using only the authoritative prefix."""

    prefix = tuple(generated_prefix)
    budget = min(MAX_DRAFT, len(prefix))  # public SuffixDecoding default: alpha=1
    compatible = [item for item in history if item.tokens[: len(prefix)] == prefix]
    result: list[int] = []
    cursor = len(prefix)
    while compatible and len(result) < budget:
        votes = Counter(
            item.tokens[cursor] for item in compatible if cursor < len(item.tokens)
        )
        if not votes:
            break
        recency = {
            token: max(
                item.order
                for item in compatible
                if cursor < len(item.tokens) and item.tokens[cursor] == token
            )
            for token in votes
        }
        token = min(votes, key=lambda value: (-votes[value], -recency[value], value))
        if votes[token] / len(compatible) < MIN_TOKEN_PROBABILITY:
            break
        result.append(token)
        compatible = [
            item
            for item in compatible
            if cursor < len(item.tokens) and item.tokens[cursor] == token
        ]
        cursor += 1
    return tuple(result)


def replay_action(actual: Sequence[int], history: Sequence[Occurrence]) -> Replay:
    actual = tuple(actual)
    generated = steps = proposals = proposed = accepted = maximum = 0
    while generated < len(actual):
        draft = trie_proposal(history, actual[:generated])
        if not draft:
            generated += 1
            steps += 1
            continue
        matched = 0
        for candidate, target in zip(draft, actual[generated:]):
            if candidate != target:
                break
            matched += 1
        proposals += 1
        proposed += len(draft)
        accepted += matched
        maximum = max(maximum, len(draft))
        steps += 1
        generated += matched
        if generated < len(actual):
            generated += 1  # verifier bonus token
    return Replay(steps, proposals, proposed, accepted, maximum)


def analyze(recording: Mapping[str, Any], tokenizer: Any) -> dict[str, Any]:
    rows = []
    totals = Counter()
    maximum_history = maximum_body = maximum_proposal = 0
    for case in recording.get("cases") or ():
        history: list[Occurrence] = []
        case_totals = Counter()
        for turn in case.get("turns") or ():
            call = turn.get("call")
            actual = tuple(int(token) for token in turn.get("target_body_tokens") or ())
            if (
                not isinstance(call, Mapping)
                or not turn.get("enabled_tool_call")
                or (turn.get("main") or {}).get("truncated")
                or not actual
            ):
                continue
            replay = replay_action(actual, history)
            case_totals.update(
                actions=1,
                control=len(actual),
                treatment=replay.target_steps,
                proposals=replay.proposals,
                proposed=replay.proposed,
                accepted=replay.accepted,
            )
            maximum_proposal = max(maximum_proposal, replay.max_proposal)
            body = qwen_body(call)
            body_bytes = len(body.encode("utf-8"))
            if body_bytes <= MAX_BODY_BYTES:
                tokens = tuple(tokenizer.encode(body, add_special_tokens=False))
                history.append(Occurrence(tokens, int(turn.get("turn_index") or 0)))
                history = history[-MAX_HISTORY:]
                maximum_history = max(maximum_history, len(history))
                maximum_body = max(maximum_body, body_bytes)
        totals.update(case_totals)
        rows.append(
            {
                "case_id": case.get("case_id"),
                **case_totals,
                "target_step_ratio": (
                    case_totals["treatment"] / case_totals["control"]
                    if case_totals["control"]
                    else None
                ),
            }
        )

    cases = list(recording.get("cases") or ())
    errors = sum(bool(case.get("error")) for case in cases)
    proposed, accepted = totals["proposed"], totals["accepted"]
    ratio = totals["treatment"] / totals["control"] if totals["control"] else None
    proposal_cases = sum(row["proposals"] > 0 for row in rows)
    worst = max((row["target_step_ratio"] for row in rows if row["target_step_ratio"] is not None), default=None)
    regressions = sum(row["treatment"] > row["control"] for row in rows)
    gates = {
        "integrity": len(cases) == 12 and errors == 0 and sum(row["actions"] > 0 for row in rows) >= 8 and totals["actions"] >= 24,
        "coverage": proposal_cases >= 8 and totals["proposals"] >= 20 and proposed >= 100,
        "draft_quality": proposed > 0 and accepted / proposed >= 0.8,
        "target_work": ratio is not None and ratio <= 0.9 and totals["control"] - totals["treatment"] >= 200,
        "beats_i30": I30_TARGET_STEPS - totals["treatment"] >= 28,
        "per_episode_safety": regressions == 0 and worst is not None and worst <= 1.0,
        "bounds": maximum_history <= MAX_HISTORY and maximum_body <= MAX_BODY_BYTES and maximum_proposal <= MAX_DRAFT,
    }
    return {
        "cases": len(cases),
        "errors": errors,
        "actions": totals["actions"],
        "proposal_cases": proposal_cases,
        "proposals": totals["proposals"],
        "proposed_tokens": proposed,
        "accepted_tokens": accepted,
        "rejected_tokens": proposed - accepted,
        "acceptance_rate": accepted / proposed if proposed else 0.0,
        "control_target_steps": totals["control"],
        "treatment_target_steps": totals["treatment"],
        "target_steps_saved": totals["control"] - totals["treatment"],
        "target_step_ratio": ratio,
        "per_case_regressions": regressions,
        "worst_case_ratio": worst,
        "max_history": maximum_history,
        "max_body_bytes": maximum_body,
        "max_proposal_tokens": maximum_proposal,
        "gates": gates,
        "discovery_gate_passed": all(gates.values()),
        "per_case": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("recording", type=Path)
    args = parser.parse_args()
    digest = hashlib.sha256(args.recording.read_bytes()).hexdigest()
    if digest != RECORDING_SHA256:
        raise ValueError(f"recording SHA-256 mismatch: {digest}")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_REPOSITORY,
        revision=TOKENIZER_REVISION,
    )
    recording = json.loads(args.recording.read_text(encoding="utf-8"))
    print(json.dumps(analyze(recording, tokenizer), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
