"""Record and replay the preregistered I30 Actor-history holdout.

Formal recordings remain private.  The checked-in code fixes the Qwen3 wire
formatter, deterministic mock-tool world, causal three-action history, unique
prefix activation, and all decision gates before the holdout is generated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

try:
    from .d2_think_tape_ablation import (
        DEFAULT_TOKENIZER,
        DEFAULT_TOKENIZER_REVISION,
        LlamaServerClient,
        _canonical_json,
        _find_subsequence,
        _normalize_call,
        _repair_main_tokens,
        _request_hash,
        _sha256_file,
        _write_json_atomic,
        load_case_manifest,
        normalize_text_messages,
        parse_main_tool_call,
    )
    from .d3_actor_history_ablation import (
        HistoryOpportunity,
        ranked_policy_candidates,
        simulate_serial_drafts,
    )
except ImportError:  # direct script execution
    from d2_think_tape_ablation import (  # type: ignore[no-redef]
        DEFAULT_TOKENIZER,
        DEFAULT_TOKENIZER_REVISION,
        LlamaServerClient,
        _canonical_json,
        _find_subsequence,
        _normalize_call,
        _repair_main_tokens,
        _request_hash,
        _sha256_file,
        _write_json_atomic,
        load_case_manifest,
        normalize_text_messages,
        parse_main_tool_call,
    )
    from d3_actor_history_ablation import (  # type: ignore[no-redef]
        HistoryOpportunity,
        ranked_policy_candidates,
        simulate_serial_drafts,
    )


RECORDING_FORMAT = "qwen-actor-history-holdout"
RECORDING_VERSION = 1
HISTORY_CAP = 3
MIN_HISTORY_CANDIDATES = 3
MIN_GENERATED_TOKENS = 1
MAX_CANDIDATES = 8
MAX_DRAFT_TOKENS = 28
MAX_BODY_BYTES = 4_096
MAX_TURNS = 6
MAX_TOKENS_PER_TURN = 512
SEED = 42
BOUNDARY = "<tool_call>"


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def qwen3_tool_body(call: Mapping[str, Any]) -> str:
    """Serialize K(a) exactly as the frozen Qwen3 tagged-JSON body."""

    normalized = _normalize_call(call)
    if normalized is None:
        raise ValueError("tool call must contain a non-empty name")
    return "\n" + json.dumps(
        {
            "name": normalized["name"],
            "arguments": normalized["arguments"],
        },
        ensure_ascii=False,
        separators=(", ", ": "),
    ) + "\n"


def action_key(call: Mapping[str, Any]) -> str:
    normalized = _normalize_call(call)
    if normalized is None:
        raise ValueError("tool call must contain a non-empty name")
    return _canonical_json(normalized)


def deterministic_tool_result(call: Mapping[str, Any]) -> str:
    """Return a bounded, side-effect-free result using only the current K(a)."""

    normalized = _normalize_call(call)
    if normalized is None:
        raise ValueError("tool call must contain a non-empty name")
    name = normalized["name"]
    arguments = normalized["arguments"]
    canonical = _canonical_json(arguments)
    digest = _sha256_text(canonical)[:16]
    path = str(arguments.get("path") or "<unspecified>")
    if name == "read":
        return (
            f"MOCK_READ path={path}\n"
            "The requested file exists in this deterministic fixture.\n"
            "Relevant source contains the issue described by the user and a nearby test.\n"
            "Continue with read, edit, write, or bash as needed."
        )
    if name == "bash":
        command = str(arguments.get("command") or "")
        return (
            "MOCK_BASH exit_code=0\n"
            f"command_sha256={_sha256_text(command)[:16]}\n"
            "stdout=mock command completed successfully"
        )
    if name == "edit":
        edits = arguments.get("edits")
        count = len(edits) if isinstance(edits, list) else 0
        return f"MOCK_EDIT applied path={path} edits={count} arguments_sha256={digest}"
    if name == "write":
        content = str(arguments.get("content") or "")
        return (
            f"MOCK_WRITE path={path} content_utf8_bytes={len(content.encode('utf-8'))} "
            f"arguments_sha256={digest}"
        )
    return f"MOCK_TOOL name={name} arguments_sha256={digest} status=ok"


def _json_type_matches(value: Any, expected: Any) -> bool:
    names = expected if isinstance(expected, list) else [expected]
    for name in names:
        if name == "object" and isinstance(value, Mapping):
            return True
        if name == "array" and isinstance(value, list):
            return True
        if name == "string" and isinstance(value, str):
            return True
        if name == "integer" and isinstance(value, int) and not isinstance(value, bool):
            return True
        if name == "number" and isinstance(value, (int, float)) and not isinstance(value, bool):
            return True
        if name == "boolean" and isinstance(value, bool):
            return True
        if name == "null" and value is None:
            return True
    return False


def enabled_tool_call(
    call: Mapping[str, Any],
    tools: Sequence[Mapping[str, Any]],
) -> bool:
    """Validate the enabled tool plus the schema subset used by the manifest."""

    normalized = _normalize_call(call)
    if normalized is None:
        return False
    definition = next(
        (
            tool.get("function")
            for tool in tools
            if isinstance(tool.get("function"), Mapping)
            and str(tool["function"].get("name") or "") == normalized["name"]
        ),
        None,
    )
    if not isinstance(definition, Mapping):
        return False
    schema = definition.get("parameters") or {}
    if not isinstance(schema, Mapping):
        return False
    arguments = normalized["arguments"]
    required = schema.get("required") or []
    if not isinstance(required, list) or any(key not in arguments for key in required):
        return False
    properties = schema.get("properties") or {}
    if not isinstance(properties, Mapping):
        return False
    for key, value in arguments.items():
        property_schema = properties.get(key)
        if not isinstance(property_schema, Mapping):
            continue
        expected = property_schema.get("type")
        if expected is not None and not _json_type_matches(value, expected):
            return False
    return True


def _last_subsequence(values: Sequence[int], needle: Sequence[int]) -> int | None:
    first = _find_subsequence(values, needle)
    if first is None:
        return None
    result = first
    offset = first + 1
    while offset <= len(values) - len(needle):
        relative = _find_subsequence(values[offset:], needle)
        if relative is None:
            break
        result = offset + relative
        offset = result + 1
    return result


def target_body_tokens(
    main_token_ids: Sequence[int],
    boundary_token_ids: Sequence[int],
) -> tuple[int, ...] | None:
    index = _last_subsequence(main_token_ids, boundary_token_ids)
    if index is None:
        return None
    start = index + len(boundary_token_ids)
    body = tuple(int(token) for token in main_token_ids[start:])
    return body or None


def _assistant_message(call: Mapping[str, Any]) -> dict[str, Any]:
    normalized = _normalize_call(call)
    if normalized is None:
        raise ValueError("tool call must contain a non-empty name")
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": normalized["name"],
                    "arguments": normalized["arguments"],
                },
            }
        ],
    }


def _tool_message(call: Mapping[str, Any], result: str) -> dict[str, Any]:
    normalized = _normalize_call(call)
    if normalized is None:
        raise ValueError("tool call must contain a non-empty name")
    return {
        "role": "tool",
        "name": normalized["name"],
        "content": result,
    }


def frozen_config(*, manifest: Path, tokenizer: Any) -> dict[str, Any]:
    return {
        "manifest_name": manifest.name,
        "manifest_sha256": _sha256_file(manifest),
        "tokenizer": DEFAULT_TOKENIZER,
        "tokenizer_revision": DEFAULT_TOKENIZER_REVISION,
        "tokenizer_resolved_revision": tokenizer.init_kwargs.get("_commit_hash"),
        "enable_thinking": False,
        "temperature": 0.0,
        "seed": SEED,
        "max_turns": MAX_TURNS,
        "max_tokens_per_turn": MAX_TOKENS_PER_TURN,
        "boundary": BOUNDARY,
        "history_cap": HISTORY_CAP,
        "min_history_candidates": MIN_HISTORY_CANDIDATES,
        "min_generated_tokens": MIN_GENERATED_TOKENS,
        "max_candidates": MAX_CANDIDATES,
        "max_draft_tokens": MAX_DRAFT_TOKENS,
        "max_body_bytes": MAX_BODY_BYTES,
        "history_activation": "shortest-unique-prefix",
        "history_formatter": "qwen3-spaced-tagged-json",
        "mock_world": "deterministic-k(a)-v1",
    }


def record_case(
    client: LlamaServerClient,
    tokenizer: Any,
    case: Mapping[str, Any],
) -> dict[str, Any]:
    messages = normalize_text_messages(case["messages"])
    tools = [dict(tool) for tool in case["tools"]]
    boundary_tokens = tuple(
        int(token)
        for token in tokenizer.encode(BOUNDARY, add_special_tokens=False)
    )
    turns = []
    stop_reason = "max_turns"
    for turn_index in range(MAX_TURNS):
        prompt = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        main = client.stream_main(
            prompt,
            max_tokens=MAX_TOKENS_PER_TURN,
            seed=SEED,
        )
        _repair_main_tokens(tokenizer, main)
        call = parse_main_tool_call(str(main.get("text") or ""))
        body_tokens = target_body_tokens(main["token_ids"], boundary_tokens)
        valid = bool(
            call
            and enabled_tool_call(call, tools)
            and body_tokens
            and not main.get("truncated")
        )
        turn = {
            "turn_index": turn_index,
            "prompt": prompt,
            "prompt_sha256": _sha256_text(prompt),
            "prompt_tokens": len(
                tokenizer.encode(prompt, add_special_tokens=False)
            ),
            "main": main,
            "call": call,
            "enabled_tool_call": bool(call and enabled_tool_call(call, tools)),
            "target_body_tokens": list(body_tokens or ()),
        }
        turns.append(turn)
        if main.get("truncated"):
            stop_reason = "truncated"
            break
        if call is None:
            stop_reason = "no_tool_call"
            break
        if not enabled_tool_call(call, tools):
            stop_reason = "invalid_or_disabled_tool_call"
            break
        if not body_tokens:
            stop_reason = "missing_token_boundary"
            break
        if not valid:  # defensive: all invalid branches above are explicit
            stop_reason = "invalid_action"
            break
        tool_result = deterministic_tool_result(call)
        turn["tool_result"] = tool_result
        messages.extend((_assistant_message(call), _tool_message(call, tool_result)))

    return {
        "case_id": case.get("case_id"),
        "request_hash": case["request_hash"],
        "sources": case["sources"],
        "turns": turns,
        "stop_reason": stop_reason,
    }


def _history_body(tokenizer: Any, call: Mapping[str, Any]) -> dict[str, Any]:
    body = qwen3_tool_body(call)
    return {
        "key": action_key(call),
        "tokens": tuple(
            int(token)
            for token in tokenizer.encode(body, add_special_tokens=False)
        ),
        "bytes": len(body.encode("utf-8")),
    }


def analyze_recording(recording: Mapping[str, Any], tokenizer: Any) -> dict[str, Any]:
    started = time.process_time_ns()
    case_rows = []
    total_actions = eligible_turns = history_candidates = 0
    control_steps = treatment_steps = proposals = proposed = accepted = 0
    proposal_cases = exact_recurrences = 0
    max_body_bytes = max_total_candidates = max_history_candidates = 0

    for case in recording.get("cases") or []:
        history: list[dict[str, Any]] = []
        case_control = case_treatment = case_proposals = 0
        case_proposed = case_accepted = case_eligible = case_actions = 0
        case_exact = 0
        for turn in case.get("turns") or []:
            call = _normalize_call(turn.get("call"))
            actual = tuple(int(token) for token in turn.get("target_body_tokens") or ())
            if (
                call is None
                or not bool(turn.get("enabled_tool_call"))
                or bool((turn.get("main") or {}).get("truncated"))
                or not actual
            ):
                continue
            case_actions += 1
            current = _history_body(tokenizer, call)
            opportunity = HistoryOpportunity(
                tape=str(case.get("case_id") or case.get("request_hash")),
                sequence=int(turn.get("turn_index") or 0),
                actual_tokens=actual,
                semantic_candidates=(),
                historical_candidates=tuple(
                    tuple(item["tokens"]) for item in history[:HISTORY_CAP]
                ),
            )
            ranked = ranked_policy_candidates(
                opportunity,
                history_cap=HISTORY_CAP,
                min_history_candidates=MIN_HISTORY_CANDIDATES,
                min_generated_tokens=MIN_GENERATED_TOKENS,
                max_candidates=MAX_CANDIDATES,
                limit=MAX_DRAFT_TOKENS,
            )
            selected_history = sum(item.source == "actor-history" for item in ranked)
            max_history_candidates = max(max_history_candidates, selected_history)
            max_total_candidates = max(max_total_candidates, len(ranked))
            case_eligible += selected_history == HISTORY_CAP
            simulation = simulate_serial_drafts(
                actual,
                ranked,
                limit=MAX_DRAFT_TOKENS,
            )
            case_control += len(actual)
            case_treatment += simulation.target_steps
            case_proposals += simulation.history_proposals
            case_proposed += simulation.history_proposed_tokens
            case_accepted += simulation.history_accepted_tokens
            if any(item["key"] == current["key"] for item in history[:HISTORY_CAP]):
                case_exact += 1

            if int(current["bytes"]) <= MAX_BODY_BYTES:
                max_body_bytes = max(max_body_bytes, int(current["bytes"]))
                history = [
                    current,
                    *(item for item in history if item["key"] != current["key"]),
                ][:HISTORY_CAP]

        total_actions += case_actions
        eligible_turns += case_eligible
        history_candidates += case_eligible * HISTORY_CAP
        control_steps += case_control
        treatment_steps += case_treatment
        proposals += case_proposals
        proposed += case_proposed
        accepted += case_accepted
        exact_recurrences += case_exact
        proposal_cases += case_proposals > 0
        case_rows.append(
            {
                "case_id": case.get("case_id"),
                "actions": case_actions,
                "eligible_turns": case_eligible,
                "exact_history_recurrences": case_exact,
                "history_proposals": case_proposals,
                "history_proposed_tokens": case_proposed,
                "history_accepted_tokens": case_accepted,
                "history_rejected_tokens": case_proposed - case_accepted,
                "control_target_steps": case_control,
                "treatment_target_steps": case_treatment,
                "target_steps_saved": case_control - case_treatment,
                "target_step_ratio": (
                    case_treatment / case_control if case_control else None
                ),
            }
        )

    cases = list(recording.get("cases") or [])
    errors = [case for case in cases if case.get("error")]
    action_cases = sum(row["actions"] > 0 for row in case_rows)
    four_action_cases = sum(row["actions"] >= 4 for row in case_rows)
    rejected = proposed - accepted
    ratio = treatment_steps / control_steps if control_steps else None
    per_case_regressions = sum(
        row["treatment_target_steps"] > row["control_target_steps"]
        for row in case_rows
    )
    worst_case_ratio = max(
        (
            float(row["target_step_ratio"])
            for row in case_rows
            if row["target_step_ratio"] is not None
        ),
        default=None,
    )
    expected_cases = int((recording.get("manifest") or {}).get("case_count") or 0)
    gates = {
        "integrity": (
            expected_cases == 12
            and len(cases) == expected_cases
            and not errors
            and action_cases >= 8
            and total_actions >= 24
        ),
        "testability": four_action_cases >= 6 and eligible_turns >= 8,
        "actual_proposals": (
            proposals >= 4 and proposed >= 28 and proposal_cases >= 3
        ),
        "draft_quality": (
            proposed > 0
            and accepted / proposed >= 0.80
            and rejected / proposed <= 0.20
        ),
        "target_work": (
            ratio is not None
            and ratio <= 0.95
            and control_steps - treatment_steps >= 28
        ),
        "per_episode_safety": (
            per_case_regressions == 0
            and worst_case_ratio is not None
            and worst_case_ratio <= 1.0
        ),
        "bounds": (
            max_body_bytes <= MAX_BODY_BYTES
            and max_history_candidates <= HISTORY_CAP
            and max_total_candidates <= MAX_CANDIDATES
        ),
    }
    return {
        "cases": len(cases),
        "case_errors": len(errors),
        "action_cases": action_cases,
        "four_action_cases": four_action_cases,
        "actions": total_actions,
        "eligible_turns": eligible_turns,
        "registered_history_candidates": history_candidates,
        "exact_history_recurrences": exact_recurrences,
        "proposal_cases": proposal_cases,
        "history_proposals": proposals,
        "history_proposed_tokens": proposed,
        "history_accepted_tokens": accepted,
        "history_rejected_tokens": rejected,
        "history_acceptance_rate": accepted / proposed if proposed else 0.0,
        "control_target_steps": control_steps,
        "treatment_target_steps": treatment_steps,
        "target_steps_saved": control_steps - treatment_steps,
        "target_step_ratio": ratio,
        "per_case_regressions": per_case_regressions,
        "worst_case_target_step_ratio": worst_case_ratio,
        "max_registered_body_bytes": max_body_bytes,
        "max_history_candidates": max_history_candidates,
        "max_total_candidates": max_total_candidates,
        "analysis_cpu_ms": (time.process_time_ns() - started) / 1_000_000,
        "gates": gates,
        "product_gate_passed": all(gates.values()),
        "per_case": case_rows,
    }


def _record(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        revision=args.tokenizer_revision,
    )
    cases, manifests = load_case_manifest(args.case_manifest)
    config = frozen_config(manifest=args.case_manifest, tokenizer=tokenizer)
    if args.tokenizer != DEFAULT_TOKENIZER:
        raise ValueError(f"formal recorder requires tokenizer {DEFAULT_TOKENIZER}")
    if args.tokenizer_revision != DEFAULT_TOKENIZER_REVISION:
        raise ValueError(
            f"formal recorder requires tokenizer revision {DEFAULT_TOKENIZER_REVISION}"
        )

    if args.output.exists():
        if not args.resume:
            raise FileExistsError("output exists; pass --resume instead of overwriting")
        recording = json.loads(args.output.read_text(encoding="utf-8"))
        if (
            recording.get("format") != RECORDING_FORMAT
            or recording.get("version") != RECORDING_VERSION
            or recording.get("config") != config
        ):
            raise ValueError("resume recording configuration does not match")
    else:
        recording = {
            "format": RECORDING_FORMAT,
            "version": RECORDING_VERSION,
            "manifest": {
                **manifests[0],
                "case_count": len(cases),
            },
            "config": config,
            "runtime": {},
            "cases": [],
        }

    completed = {str(case.get("request_hash")) for case in recording["cases"]}
    client = LlamaServerClient(args.server, timeout_s=args.timeout_s)
    recording["runtime"]["health"] = client.health()
    for index, case in enumerate(cases, start=1):
        if case["request_hash"] in completed:
            continue
        print(f"recording {index}/{len(cases)} {case.get('case_id')}", flush=True)
        try:
            result = record_case(client, tokenizer, case)
        except Exception as error:  # preserve the one formal attempt for audit/resume
            result = {
                "case_id": case.get("case_id"),
                "request_hash": case["request_hash"],
                "sources": case["sources"],
                "turns": [],
                "stop_reason": "error",
                "error": f"{type(error).__name__}: {error}",
            }
        recording["cases"].append(result)
        recording["analysis"] = analyze_recording(recording, tokenizer)
        _write_json_atomic(args.output, recording)
    return recording


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    record = subparsers.add_parser("record")
    record.add_argument("--case-manifest", type=Path, required=True)
    record.add_argument("--output", type=Path, required=True)
    record.add_argument("--server", default="http://127.0.0.1:8080")
    record.add_argument("--timeout-s", type=float, default=600.0)
    record.add_argument("--resume", action="store_true")
    record.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    record.add_argument(
        "--tokenizer-revision",
        default=DEFAULT_TOKENIZER_REVISION,
    )
    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--recording", type=Path, required=True)
    analyze.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    analyze.add_argument(
        "--tokenizer-revision",
        default=DEFAULT_TOKENIZER_REVISION,
    )
    args = parser.parse_args()

    if args.command == "record":
        result = _record(args)
        print(json.dumps(result.get("analysis") or {}, ensure_ascii=False, indent=2))
        return

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        revision=args.tokenizer_revision,
    )
    recording = json.loads(args.recording.read_text(encoding="utf-8"))
    result = analyze_recording(recording, tokenizer)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
