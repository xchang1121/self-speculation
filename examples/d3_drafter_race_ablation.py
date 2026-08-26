"""Compare full-width Drafter candidates with first-valid hedge admission.

This is a target-verifier work proxy. Provider-side residual service is measured
by the companion Pi tape analyzer; this script asks only whether canceling the
late response removes a candidate that the serial D3 verifier would use.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

if __package__:
    from .d3_tape_ablation import (
        DraftLengthResult,
        analyze_opportunities,
        build_opportunities,
        parse_tape,
    )
else:
    from d3_tape_ablation import (  # type: ignore[no-redef]
        DraftLengthResult,
        analyze_opportunities,
        build_opportunities,
        parse_tape,
    )


def aggregate_results(
    results: Sequence[DraftLengthResult], *, policy: str
) -> dict[str, Any]:
    if not results:
        raise ValueError("at least one D3 Drafter-race result is required")
    limits = {result.max_draft_tokens for result in results}
    orderings = {result.ordering for result in results}
    if len(limits) != 1 or len(orderings) != 1:
        raise ValueError("D3 Drafter-race results must share one configuration")
    actor_tokens = sum(result.actor_tokens for result in results)
    proposed = sum(result.proposed_tokens for result in results)
    accepted = sum(result.accepted_tokens for result in results)
    target_steps = sum(result.target_steps for result in results)
    saved = actor_tokens - target_steps
    return {
        "policy": policy,
        "tapes": len(results),
        "ordering": results[0].ordering,
        "max_draft_tokens": results[0].max_draft_tokens,
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

    if args.max_draft_tokens <= 0:
        parser.error("--max-draft-tokens must be positive")
    if args.drafter_width <= 0:
        parser.error("--drafter-width must be positive")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, revision=args.revision)
    policies = {"full_width": None, "first_valid": 1}
    results_by_policy: dict[str, list[DraftLengthResult]] = {
        policy: [] for policy in policies
    }
    per_tape = []
    for tape in args.tape:
        parsed = parse_tape(tape, args.format)
        tape_results = []
        for policy, completion_limit in policies.items():
            opportunities = build_opportunities(
                parsed,
                actor_model=args.actor_model,
                drafter_model=args.drafter_model,
                tokenizer=tokenizer,
                drafter_width=args.drafter_width,
                drafter_completion_limit=completion_limit,
            )
            result = analyze_opportunities(
                opportunities,
                limits=(args.max_draft_tokens,),
            )[0]
            results_by_policy[policy].append(result)
            tape_results.append({"policy": policy, **asdict(result)})
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
                    policy: aggregate_results(results, policy=policy)
                    for policy, results in results_by_policy.items()
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
