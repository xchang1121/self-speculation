"""Ablate candidate-agreement clipping for serial D3 verification.

The treatment proposes only the shared next-token prefix while multiple
registered candidates remain compatible with the target sequence. Once the
target's bonus token selects a branch, the surviving candidate can provide its
remaining suffix. This is a verifier-work proxy, not a wall-clock claim.
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
        _common_prefix,
        _simulate,
        build_opportunities,
        parse_tape,
    )
else:
    from d3_tape_ablation import (  # type: ignore[no-redef]
        DraftOpportunity,
        _common_prefix,
        _simulate,
        build_opportunities,
        parse_tape,
    )


@dataclass(frozen=True, slots=True)
class AgreementCapSimulation:
    target_steps: int
    proposals: int
    proposed_tokens: int
    accepted_tokens: int
    clipped_proposals: int
    clipped_tokens: int


@dataclass(frozen=True, slots=True)
class AgreementCapResult:
    policy: str
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
    clipped_proposals: int
    clipped_tokens: int


def _shared_prefix_length(values: Sequence[Sequence[int]]) -> int:
    if not values:
        return 0
    shared = len(values[0])
    for value in values[1:]:
        shared = min(shared, _common_prefix(values[0][:shared], value))
        if shared == 0:
            break
    return shared


def simulate_agreement_cap(
    actual: Sequence[int],
    candidates: Sequence[Sequence[int]],
    limit: int,
) -> AgreementCapSimulation:
    """Replay a serial store that clips a proposal at candidate disagreement."""

    if limit <= 0:
        raise ValueError("limit must be positive")
    active = [tuple(candidate[:limit]) for candidate in candidates if candidate]
    generated = 0
    target_steps = 0
    proposals = 0
    proposed_tokens = 0
    accepted_tokens = 0
    clipped_proposals = 0
    clipped_tokens = 0
    actual_tuple = tuple(actual)

    while generated < len(actual_tuple):
        compatible = [
            candidate
            for candidate in active
            if generated < len(candidate)
            and actual_tuple[:generated] == candidate[:generated]
        ]
        if not compatible:
            target_steps += len(actual_tuple) - generated
            break

        primary = compatible[0]
        primary_suffix = primary[generated:]
        proposal = primary_suffix
        clipped = False
        if len(compatible) > 1:
            suffixes = [candidate[generated:] for candidate in compatible]
            shared = _shared_prefix_length(suffixes)
            if shared > 0 and shared < len(primary_suffix):
                proposal = primary_suffix[:shared]
                clipped = True

        if clipped:
            clipped_proposals += 1
            clipped_tokens += len(primary_suffix) - len(proposal)
        else:
            active.remove(primary)

        accepted = _common_prefix(proposal, actual_tuple[generated:])
        target_steps += 1
        proposals += 1
        proposed_tokens += len(proposal)
        accepted_tokens += accepted
        generated += accepted
        if generated < len(actual_tuple):
            # The verifier round also emits one authoritative target token.
            generated += 1

    return AgreementCapSimulation(
        target_steps=target_steps,
        proposals=proposals,
        proposed_tokens=proposed_tokens,
        accepted_tokens=accepted_tokens,
        clipped_proposals=clipped_proposals,
        clipped_tokens=clipped_tokens,
    )


def analyze_agreement_cap(
    opportunities: Sequence[DraftOpportunity],
    *,
    policy: str,
    max_draft_tokens: int = 28,
) -> AgreementCapResult:
    if policy not in {"full", "agreement-cap"}:
        raise ValueError(f"unsupported agreement-cap policy: {policy}")
    actor_tokens = sum(len(opportunity.actual_tokens) for opportunity in opportunities)
    candidate_count = sum(
        len(opportunity.candidate_tokens) for opportunity in opportunities
    )
    proposals = proposed = accepted = target_steps = 0
    clipped_proposals = clipped_tokens = 0
    for opportunity in opportunities:
        if policy == "full":
            steps, proposal_count, offered, matched = _simulate(
                opportunity.actual_tokens,
                opportunity.candidate_tokens,
                max_draft_tokens,
            )
        else:
            simulation = simulate_agreement_cap(
                opportunity.actual_tokens,
                opportunity.candidate_tokens,
                max_draft_tokens,
            )
            steps = simulation.target_steps
            proposal_count = simulation.proposals
            offered = simulation.proposed_tokens
            matched = simulation.accepted_tokens
            clipped_proposals += simulation.clipped_proposals
            clipped_tokens += simulation.clipped_tokens
        target_steps += steps
        proposals += proposal_count
        proposed += offered
        accepted += matched
    saved = actor_tokens - target_steps
    return AgreementCapResult(
        policy=policy,
        opportunities=len(opportunities),
        candidate_count=candidate_count,
        actor_tokens=actor_tokens,
        proposals=proposals,
        proposed_tokens=proposed,
        accepted_tokens=accepted,
        rejected_tokens=proposed - accepted,
        acceptance_rate=accepted / proposed if proposed else 0.0,
        target_steps=target_steps,
        target_steps_saved=saved,
        target_step_reduction=saved / actor_tokens if actor_tokens else 0.0,
        clipped_proposals=clipped_proposals,
        clipped_tokens=clipped_tokens,
    )


def aggregate_results(results: Sequence[AgreementCapResult]) -> dict[str, Any]:
    if not results:
        raise ValueError("at least one agreement-cap result is required")
    policies = {result.policy for result in results}
    if len(policies) != 1:
        raise ValueError("only one policy can be aggregated at a time")
    actor_tokens = sum(result.actor_tokens for result in results)
    proposed = sum(result.proposed_tokens for result in results)
    accepted = sum(result.accepted_tokens for result in results)
    target_steps = sum(result.target_steps for result in results)
    saved = actor_tokens - target_steps
    return {
        "policy": results[0].policy,
        "tapes": len(results),
        "opportunities": sum(result.opportunities for result in results),
        "candidate_count": sum(result.candidate_count for result in results),
        "actor_tokens": actor_tokens,
        "proposals": sum(result.proposals for result in results),
        "proposed_tokens": proposed,
        "accepted_tokens": accepted,
        "rejected_tokens": proposed - accepted,
        "acceptance_rate": accepted / proposed if proposed else 0.0,
        "target_steps": target_steps,
        "target_steps_saved": saved,
        "target_step_reduction": saved / actor_tokens if actor_tokens else 0.0,
        "clipped_proposals": sum(result.clipped_proposals for result in results),
        "clipped_tokens": sum(result.clipped_tokens for result in results),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tape", type=Path, action="append", required=True)
    parser.add_argument("--actor-model", required=True)
    parser.add_argument("--drafter-model", required=True)
    parser.add_argument("--tokenizer", default="deepseek-ai/DeepSeek-V3")
    parser.add_argument("--revision")
    parser.add_argument("--format", default="tagged_json")
    parser.add_argument("--max-draft-tokens", type=int, default=28)
    parser.add_argument("--drafter-width", type=int, default=2)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, revision=args.revision)
    policies = ("full", "agreement-cap")
    results_by_policy: dict[str, list[AgreementCapResult]] = {
        policy: [] for policy in policies
    }
    per_tape = []
    for tape in args.tape:
        opportunities = build_opportunities(
            parse_tape(tape, args.format),
            actor_model=args.actor_model,
            drafter_model=args.drafter_model,
            tokenizer=tokenizer,
            drafter_width=args.drafter_width,
        )
        tape_results = []
        for policy in policies:
            result = analyze_agreement_cap(
                opportunities,
                policy=policy,
                max_draft_tokens=args.max_draft_tokens,
            )
            results_by_policy[policy].append(result)
            tape_results.append(asdict(result))
        per_tape.append({"tape": str(tape), "results": tape_results})

    print(
        json.dumps(
            {
                "tokenizer": args.tokenizer,
                "tokenizer_revision": args.revision
                or tokenizer.init_kwargs.get("_commit_hash"),
                "format": args.format,
                "max_draft_tokens": args.max_draft_tokens,
                "drafter_width": args.drafter_width,
                "per_tape": per_tape,
                "pooled": {
                    policy: aggregate_results(results)
                    for policy, results in results_by_policy.items()
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
