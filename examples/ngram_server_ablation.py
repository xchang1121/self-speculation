"""Run a lossless native n-gram speculative-decoding A/B on frozen cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__:
    from .d2_think_tape_ablation import (
        DEFAULT_TOKENIZER,
        DEFAULT_TOKENIZER_REVISION,
        LlamaServerClient,
        _repair_main_tokens,
        _write_json_atomic,
        load_case_manifest,
        normalize_text_messages,
        parse_main_tool_call,
    )
else:
    from d2_think_tape_ablation import (  # type: ignore[no-redef]
        DEFAULT_TOKENIZER,
        DEFAULT_TOKENIZER_REVISION,
        LlamaServerClient,
        _repair_main_tokens,
        _write_json_atomic,
        load_case_manifest,
        normalize_text_messages,
        parse_main_tool_call,
    )


METRIC_NAMES = (
    "llamacpp:prompt_tokens_total",
    "llamacpp:prompt_tokens_cached_total",
    "llamacpp:prompt_seconds_total",
    "llamacpp:tokens_predicted_total",
    "llamacpp:tokens_predicted_seconds_total",
    "llamacpp:n_decode_total",
    "llamacpp:spec_decode_num_draft_tokens_total",
    "llamacpp:spec_decode_num_accepted_tokens_total",
    "llamacpp:spec_decode_num_drafts_total",
)
WARMUP_PROMPT = (
    "Repeat exactly: alpha beta gamma alpha beta gamma alpha beta gamma"
)


def parse_prometheus_metrics(text: str) -> dict[str, float]:
    """Extract only the fixed unlabeled llama.cpp counters used by this A/B."""

    selected = set(METRIC_NAMES)
    result: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name, separator, raw_value = line.partition(" ")
        if not separator or name not in selected:
            continue
        value = float(raw_value.strip())
        if math.isfinite(value):
            result[name] = value
    missing = selected - result.keys()
    if missing:
        raise ValueError(f"metrics response is missing counters: {sorted(missing)}")
    return result


def _metric_delta(
    before: Mapping[str, float],
    after: Mapping[str, float],
) -> dict[str, float]:
    return {name: float(after[name]) - float(before[name]) for name in METRIC_NAMES}


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator > 0.0 else None


def _sum_main(pairs: Sequence[Mapping[str, Any]], arm: str, field: str) -> float:
    return sum(float(((pair.get(arm) or {}).get(field)) or 0.0) for pair in pairs)


def _sum_timing(pairs: Sequence[Mapping[str, Any]], arm: str, field: str) -> float:
    return sum(
        float((((pair.get(arm) or {}).get("timings") or {}).get(field)) or 0.0)
        for pair in pairs
    )


def analyze_recording(recording: Mapping[str, Any]) -> dict[str, Any]:
    config = recording.get("config") or {}
    pairs = [pair for pair in recording.get("pairs") or [] if isinstance(pair, Mapping)]
    complete = [
        pair
        for pair in pairs
        if isinstance(pair.get("control"), Mapping)
        and isinstance(pair.get("treatment"), Mapping)
        and not pair.get("errors")
    ]
    expected_pairs = int(config.get("expected_pairs") or len(pairs))
    min_nonempty_pairs = int(config.get("gate_min_nonempty_pairs") or 8)
    min_draft_tokens = int(config.get("gate_min_draft_tokens") or 28)
    min_acceptance = float(config.get("gate_min_draft_acceptance") or 0.50)
    max_pooled_ratio = float(config.get("gate_max_pooled_ratio") or 0.95)
    max_tail_ratio = float(config.get("gate_max_tail_ratio") or 1.25)

    exact_pairs = [
        pair
        for pair in complete
        if pair["control"].get("token_ids") == pair["treatment"].get("token_ids")
        and pair["control"].get("text") == pair["treatment"].get("text")
        and pair["control"].get("stop_type")
        == pair["treatment"].get("stop_type")
        and bool(pair["control"].get("truncated"))
        == bool(pair["treatment"].get("truncated"))
    ]
    nonempty_pairs = sum(bool((pair["control"].get("token_ids") or [])) for pair in complete)
    control_predicted_ms = _sum_timing(complete, "control", "predicted_ms")
    treatment_predicted_ms = _sum_timing(complete, "treatment", "predicted_ms")
    control_wall_ms = _sum_main(complete, "control", "wall_ms")
    treatment_wall_ms = _sum_main(complete, "treatment", "wall_ms")
    predicted_ratios = [
        ratio
        for pair in complete
        if (
            ratio := _ratio(
                float(((pair["treatment"].get("timings") or {}).get("predicted_ms")) or 0.0),
                float(((pair["control"].get("timings") or {}).get("predicted_ms")) or 0.0),
            )
        )
        is not None
    ]

    metrics = recording.get("metrics") or {}
    control_metrics = (metrics.get("delta") or {}).get("control") or {}
    treatment_metrics = (metrics.get("delta") or {}).get("treatment") or {}
    control_forwards = float(control_metrics.get("llamacpp:n_decode_total") or 0.0) + float(
        control_metrics.get("llamacpp:spec_decode_num_drafts_total") or 0.0
    )
    treatment_forwards = float(
        treatment_metrics.get("llamacpp:n_decode_total") or 0.0
    ) + float(treatment_metrics.get("llamacpp:spec_decode_num_drafts_total") or 0.0)
    draft_tokens = float(
        treatment_metrics.get("llamacpp:spec_decode_num_draft_tokens_total") or 0.0
    )
    accepted_tokens = float(
        treatment_metrics.get("llamacpp:spec_decode_num_accepted_tokens_total") or 0.0
    )
    acceptance = _ratio(accepted_tokens, draft_tokens)
    forward_ratio = _ratio(treatment_forwards, control_forwards)
    predicted_ratio = _ratio(treatment_predicted_ms, control_predicted_ms)
    wall_ratio = _ratio(treatment_wall_ms, control_wall_ms)
    median_predicted_ratio = statistics.median(predicted_ratios) if predicted_ratios else None
    max_predicted_ratio = max(predicted_ratios) if predicted_ratios else None
    gates = {
        "validity": len(complete) == expected_pairs and nonempty_pairs >= min_nonempty_pairs,
        "losslessness": len(exact_pairs) == expected_pairs,
        "useful_speculation": draft_tokens >= min_draft_tokens
        and acceptance is not None
        and acceptance >= min_acceptance,
        "target_work": forward_ratio is not None and forward_ratio <= max_pooled_ratio,
        "decode_time": predicted_ratio is not None
        and predicted_ratio <= max_pooled_ratio
        and wall_ratio is not None
        and wall_ratio <= max_pooled_ratio
        and median_predicted_ratio is not None
        and median_predicted_ratio <= max_pooled_ratio,
        "tail_safety": len(predicted_ratios) == expected_pairs
        and max_predicted_ratio is not None
        and max_predicted_ratio <= max_tail_ratio,
    }
    return {
        "expected_pairs": expected_pairs,
        "recorded_pairs": len(pairs),
        "complete_pairs": len(complete),
        "nonempty_pairs": nonempty_pairs,
        "token_exact_pairs": len(exact_pairs),
        "parseable_tool_calls": {
            arm: sum((pair[arm].get("call") is not None) for pair in complete)
            for arm in ("control", "treatment")
        },
        "truncated_pairs": {
            arm: sum(bool(pair[arm].get("truncated")) for pair in complete)
            for arm in ("control", "treatment")
        },
        "generated_tokens": {
            arm: int(_sum_timing(complete, arm, "predicted_n"))
            for arm in ("control", "treatment")
        },
        "native_speculation": {
            "draft_tokens": int(draft_tokens),
            "accepted_tokens": int(accepted_tokens),
            "acceptance_rate": acceptance,
            "verification_steps": int(
                treatment_metrics.get("llamacpp:spec_decode_num_drafts_total") or 0
            ),
        },
        "target_forward_proxy": {
            "control": int(control_forwards),
            "treatment": int(treatment_forwards),
            "ratio": forward_ratio,
        },
        "predicted_ms": {
            "control": control_predicted_ms,
            "treatment": treatment_predicted_ms,
            "ratio": predicted_ratio,
            "median_paired_ratio": median_predicted_ratio,
            "max_paired_ratio": max_predicted_ratio,
        },
        "wall_ms": {
            "control": control_wall_ms,
            "treatment": treatment_wall_ms,
            "ratio": wall_ratio,
        },
        "per_case": [
            {
                "case_id": pair.get("case_id"),
                "order": pair.get("order"),
                "tokens": int(((pair["control"].get("timings") or {}).get("predicted_n")) or 0),
                "draft_tokens": int(
                    ((pair["treatment"].get("timings") or {}).get("draft_n")) or 0
                ),
                "accepted_tokens": int(
                    ((pair["treatment"].get("timings") or {}).get("draft_n_accepted"))
                    or 0
                ),
                "predicted_ms_ratio": _ratio(
                    float(
                        ((pair["treatment"].get("timings") or {}).get("predicted_ms"))
                        or 0.0
                    ),
                    float(
                        ((pair["control"].get("timings") or {}).get("predicted_ms"))
                        or 0.0
                    ),
                ),
                "wall_ms_ratio": _ratio(
                    float(pair["treatment"].get("wall_ms") or 0.0),
                    float(pair["control"].get("wall_ms") or 0.0),
                ),
            }
            for pair in complete
        ],
        "gates": gates,
        "product_gate_passed": all(gates.values()),
    }


def _http_get_json(base_url: str, path: str, timeout_s: float) -> dict[str, Any]:
    import httpx

    response = httpx.get(f"{base_url.rstrip('/')}{path}", timeout=timeout_s)
    response.raise_for_status()
    return dict(response.json())


def _metrics(base_url: str, timeout_s: float) -> dict[str, float]:
    import httpx

    response = httpx.get(f"{base_url.rstrip('/')}/metrics", timeout=timeout_s)
    response.raise_for_status()
    return parse_prometheus_metrics(response.text)


def _server_metadata(base_url: str, timeout_s: float) -> dict[str, Any]:
    props = _http_get_json(base_url, "/props", timeout_s)
    settings = props.get("default_generation_settings") or {}
    return {
        "build_info": props.get("build_info"),
        "model_alias": props.get("model_alias"),
        "model_ftype": props.get("model_ftype"),
        "n_ctx": settings.get("n_ctx"),
        "total_slots": props.get("total_slots"),
    }


def _run_main(
    client: LlamaServerClient,
    tokenizer: Any,
    case: Mapping[str, Any],
    *,
    max_tokens: int,
    seed: int,
) -> dict[str, Any]:
    prompt = tokenizer.apply_chat_template(
        normalize_text_messages(case["messages"]),
        tools=case["tools"],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    main = client.stream_main(prompt, max_tokens=max_tokens, seed=seed)
    _repair_main_tokens(tokenizer, main)
    main["call"] = parse_main_tool_call(main["text"])
    return main


def _warmup(client: LlamaServerClient, *, seed: int) -> dict[str, Any]:
    return client.stream_main(WARMUP_PROMPT, max_tokens=64, seed=seed)


def _record(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    cases, _ = load_case_manifest(args.case_manifest)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, revision=args.revision)
    clients = {
        "control": LlamaServerClient(args.control_url, timeout_s=args.timeout_s),
        "treatment": LlamaServerClient(args.treatment_url, timeout_s=args.timeout_s),
    }
    for client in clients.values():
        client.health()
    server_metadata = {
        arm: _server_metadata(url, args.timeout_s)
        for arm, url in (
            ("control", args.control_url),
            ("treatment", args.treatment_url),
        )
    }
    if server_metadata["control"] != server_metadata["treatment"]:
        raise ValueError("control and treatment server metadata do not match")

    warmup = {arm: _warmup(client, seed=args.seed) for arm, client in clients.items()}
    if warmup["control"].get("token_ids") != warmup["treatment"].get("token_ids"):
        raise ValueError("excluded warm-up outputs do not match")
    if int((warmup["control"].get("timings") or {}).get("draft_n") or 0) != 0:
        raise ValueError("control server unexpectedly drafted during warm-up")
    if int((warmup["treatment"].get("timings") or {}).get("draft_n") or 0) <= 0:
        raise ValueError("treatment server did not draft during warm-up")

    before = {
        arm: _metrics(url, args.timeout_s)
        for arm, url in (
            ("control", args.control_url),
            ("treatment", args.treatment_url),
        )
    }
    manifest_bytes = args.case_manifest.read_bytes()
    recording: dict[str, Any] = {
        "format": "self-speculation-native-ngram-ab",
        "version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "config": {
            "case_manifest": str(args.case_manifest),
            "case_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "expected_pairs": len(cases),
            "control_url": args.control_url,
            "treatment_url": args.treatment_url,
            "tokenizer": args.tokenizer,
            "tokenizer_revision": args.revision,
            "main_max_tokens": args.main_max_tokens,
            "seed": args.seed,
            "alternating_pair_order": True,
            "warmup_prompt_sha256": hashlib.sha256(WARMUP_PROMPT.encode()).hexdigest(),
            "gate_min_nonempty_pairs": 8,
            "gate_min_draft_tokens": 28,
            "gate_min_draft_acceptance": 0.50,
            "gate_max_pooled_ratio": 0.95,
            "gate_max_tail_ratio": 1.25,
        },
        "servers": server_metadata,
        "excluded_warmup": {
            arm: {
                "token_ids": value.get("token_ids"),
                "timings": value.get("timings"),
            }
            for arm, value in warmup.items()
        },
        "metrics": {"before": before},
        "pairs": [],
    }

    for index, case in enumerate(cases):
        order = ("control", "treatment") if index % 2 == 0 else ("treatment", "control")
        print(f"recording {index + 1}/{len(cases)} {case.get('case_id')}", flush=True)
        pair: dict[str, Any] = {
            "case_id": case.get("case_id"),
            "request_hash": case["request_hash"],
            "order": list(order),
        }
        errors = {}
        for arm in order:
            try:
                pair[arm] = _run_main(
                    clients[arm],
                    tokenizer,
                    case,
                    max_tokens=args.main_max_tokens,
                    seed=args.seed,
                )
            except Exception as error:  # preserve the other arm and later pairs
                errors[arm] = f"{type(error).__name__}: {error}"
        if errors:
            pair["errors"] = errors
        recording["pairs"].append(pair)
        after = {
            arm: _metrics(url, args.timeout_s)
            for arm, url in (
                ("control", args.control_url),
                ("treatment", args.treatment_url),
            )
        }
        recording["metrics"]["after"] = after
        recording["metrics"]["delta"] = {
            arm: _metric_delta(before[arm], after[arm]) for arm in clients
        }
        recording["analysis"] = analyze_recording(recording)
        _write_json_atomic(args.output, recording)
    return recording


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    record = subparsers.add_parser("record")
    record.add_argument("--case-manifest", type=Path, required=True)
    record.add_argument("--output", type=Path, required=True)
    record.add_argument("--control-url", default="http://127.0.0.1:18080")
    record.add_argument("--treatment-url", default="http://127.0.0.1:18081")
    record.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    record.add_argument("--revision", default=DEFAULT_TOKENIZER_REVISION)
    record.add_argument("--main-max-tokens", type=int, default=3072)
    record.add_argument("--seed", type=int, default=42)
    record.add_argument("--timeout-s", type=float, default=600.0)

    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--recording", type=Path, required=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "record":
        if args.main_max_tokens <= 0 or args.timeout_s <= 0:
            raise SystemExit("token limit and timeout must be positive")
        recording = _record(args)
        result = {"output": str(args.output), "analysis": recording["analysis"]}
    else:
        recording = json.loads(args.recording.read_text(encoding="utf-8"))
        result = analyze_recording(recording)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
