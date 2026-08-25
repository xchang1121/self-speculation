"""Replay a causal request-level D3 cap controller on recorded tool calls.

The controller observes only completed target verification from earlier Actor
requests. It never consults a future Actor action. Results remain verifier-work
proxies because the tapes contain no target verification wall time.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

if __package__:
    from .d3_tape_ablation import (
        DraftOpportunity,
        _simulate,
        build_opportunities,
        parse_tape,
    )
else:
    from d3_tape_ablation import (  # type: ignore[no-redef]
        DraftOpportunity,
        _simulate,
        build_opportunities,
        parse_tape,
    )


@dataclass(frozen=True, slots=True)
class DynamicCapObservation:
    sequence: int
    cap: int
    proposals: int
    proposed_tokens: int
    accepted_tokens: int
    rejected_tokens: int
    target_steps: int


@dataclass(frozen=True, slots=True)
class DynamicCapResult:
    policy: str
    opportunities: int
    actor_tokens: int
    proposals: int
    proposed_tokens: int
    accepted_tokens: int
    rejected_tokens: int
    acceptance_rate: float
    target_steps: int
    target_steps_saved: int
    target_step_reduction: float
    initial_cap: int
    final_cap: int
    mean_cap: float
    cap_trace: tuple[int, ...]
    observations: tuple[DynamicCapObservation, ...]


def simulate_dynamic_cap(
    opportunities: Sequence[DraftOpportunity],
    *,
    policy: str,
    initial_cap: int = 28,
    min_cap: int = 4,
    max_cap: int = 28,
) -> DynamicCapResult:
    """Simulate one cold-start policy in strict Actor sequence order.

    ``hf-heuristic`` is the request-level form of Transformers' documented
    persistent schedule: increase lookahead by two after complete acceptance,
    otherwise decrease it by one. A request with no offer supplies no signal.
    """

    if min_cap <= 0 or max_cap < min_cap:
        raise ValueError("dynamic cap bounds are invalid")
    if initial_cap < min_cap or initial_cap > max_cap:
        raise ValueError("initial_cap must be within the configured bounds")
    if policy not in {"fixed", "hf-heuristic"}:
        raise ValueError(f"unsupported dynamic cap policy: {policy}")

    cap = initial_cap
    cap_trace = []
    observations = []
    proposals = proposed = accepted = target_steps = 0
    ordered = sorted(opportunities, key=lambda opportunity: opportunity.sequence)
    for opportunity in ordered:
        cap_trace.append(cap)
        steps, proposal_count, offered, matched = _simulate(
            opportunity.actual_tokens,
            opportunity.candidate_tokens,
            cap,
        )
        target_steps += steps
        proposals += proposal_count
        proposed += offered
        accepted += matched
        observations.append(
            DynamicCapObservation(
                sequence=opportunity.sequence,
                cap=cap,
                proposals=proposal_count,
                proposed_tokens=offered,
                accepted_tokens=matched,
                rejected_tokens=offered - matched,
                target_steps=steps,
            )
        )
        if policy == "hf-heuristic" and offered > 0:
            cap = (
                min(max_cap, cap + 2)
                if matched == offered
                else max(min_cap, cap - 1)
            )

    actor_tokens = sum(len(opportunity.actual_tokens) for opportunity in ordered)
    saved = actor_tokens - target_steps
    return DynamicCapResult(
        policy=policy,
        opportunities=len(ordered),
        actor_tokens=actor_tokens,
        proposals=proposals,
        proposed_tokens=proposed,
        accepted_tokens=accepted,
        rejected_tokens=proposed - accepted,
        acceptance_rate=accepted / proposed if proposed else 0.0,
        target_steps=target_steps,
        target_steps_saved=saved,
        target_step_reduction=saved / actor_tokens if actor_tokens else 0.0,
        initial_cap=initial_cap,
        final_cap=cap,
        mean_cap=sum(cap_trace) / len(cap_trace) if cap_trace else float(initial_cap),
        cap_trace=tuple(cap_trace),
        observations=tuple(observations),
    )


def aggregate_results(
    results: Sequence[DynamicCapResult],
) -> dict[str, Any]:
    if not results:
        raise ValueError("at least one dynamic-cap result is required")
    policies = {result.policy for result in results}
    if len(policies) != 1:
        raise ValueError("only one policy can be aggregated at a time")
    opportunities = sum(result.opportunities for result in results)
    actor_tokens = sum(result.actor_tokens for result in results)
    proposed = sum(result.proposed_tokens for result in results)
    accepted = sum(result.accepted_tokens for result in results)
    target_steps = sum(result.target_steps for result in results)
    saved = actor_tokens - target_steps
    return {
        "policy": results[0].policy,
        "tapes": len(results),
        "opportunities": opportunities,
        "actor_tokens": actor_tokens,
        "proposals": sum(result.proposals for result in results),
        "proposed_tokens": proposed,
        "accepted_tokens": accepted,
        "rejected_tokens": proposed - accepted,
        "acceptance_rate": accepted / proposed if proposed else 0.0,
        "target_steps": target_steps,
        "target_steps_saved": saved,
        "target_step_reduction": saved / actor_tokens if actor_tokens else 0.0,
        "mean_cap": (
            sum(result.mean_cap * result.opportunities for result in results)
            / opportunities
            if opportunities
            else 0.0
        ),
    }


def _opportunities(
    tape: Path,
    *,
    actor_model: str,
    drafter_model: str,
    tokenizer: Any,
    format_name: str,
) -> tuple[DraftOpportunity, ...]:
    return build_opportunities(
        parse_tape(tape, format_name),
        actor_model=actor_model,
        drafter_model=drafter_model,
        tokenizer=tokenizer,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tape", type=Path, action="append", required=True)
    parser.add_argument("--actor-model", required=True)
    parser.add_argument("--drafter-model", required=True)
    parser.add_argument("--tokenizer", default="deepseek-ai/DeepSeek-V3")
    parser.add_argument("--revision")
    parser.add_argument("--format", default="tagged_json")
    parser.add_argument("--initial-cap", type=int, default=28)
    parser.add_argument("--min-cap", type=int, default=4)
    parser.add_argument("--max-cap", type=int, default=28)
    parser.add_argument("--policies", default="fixed,hf-heuristic")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        revision=args.revision,
    )
    policies = tuple(
        policy.strip() for policy in args.policies.split(",") if policy.strip()
    )
    per_tape = []
    pooled: dict[str, Any] = {}
    results_by_policy: dict[str, list[DynamicCapResult]] = {
        policy: [] for policy in policies
    }
    for tape in args.tape:
        opportunities = _opportunities(
            tape,
            actor_model=args.actor_model,
            drafter_model=args.drafter_model,
            tokenizer=tokenizer,
            format_name=args.format,
        )
        tape_results = []
        for policy in policies:
            result = simulate_dynamic_cap(
                opportunities,
                policy=policy,
                initial_cap=args.initial_cap,
                min_cap=args.min_cap,
                max_cap=args.max_cap,
            )
            results_by_policy[policy].append(result)
            tape_results.append(asdict(result))
        per_tape.append({"tape": str(tape), "results": tape_results})
    for policy, results in results_by_policy.items():
        pooled[policy] = aggregate_results(results)

    print(
        json.dumps(
            {
                "tokenizer": args.tokenizer,
                "tokenizer_revision": args.revision
                or tokenizer.init_kwargs.get("_commit_hash"),
                "format": args.format,
                "cold_start_per_tape": True,
                "per_tape": per_tape,
                "pooled": pooled,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
