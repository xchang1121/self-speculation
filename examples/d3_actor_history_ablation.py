"""Replay causal Actor-action history as a low-priority D3 fallback.

This analyzer deliberately keeps semantic candidates (Drafter, PatternAware,
or self-fork) ahead of retrieval candidates.  Historical actions become
eligible only after the target body reaches its shortest prefix that is unique
within the retained history group, so an ambiguous history item is never
proposed at the raw tool-call boundary.

The replay is a target-forward proxy, not a wall-clock claim.  It uses only
actions that preceded the current Actor action in the same tape.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    from .d3_tape_ablation import (
        DraftOpportunity,
        build_opportunities,
        parse_tape,
    )
except ImportError:  # direct script execution
    from d3_tape_ablation import (  # type: ignore[no-redef]
        DraftOpportunity,
        build_opportunities,
        parse_tape,
    )


@dataclass(frozen=True, slots=True)
class RankedDraft:
    tokens: tuple[int, ...]
    source: str
    min_generated_tokens: int = 0


@dataclass(frozen=True, slots=True)
class HistoryOpportunity:
    tape: str
    sequence: int
    actual_tokens: tuple[int, ...]
    semantic_candidates: tuple[tuple[int, ...], ...]
    historical_candidates: tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True)
class Simulation:
    target_steps: int
    proposals: int
    proposed_tokens: int
    accepted_tokens: int
    history_proposals: int
    history_proposed_tokens: int
    history_accepted_tokens: int

    @property
    def rejected_tokens(self) -> int:
        return self.proposed_tokens - self.accepted_tokens

    @property
    def history_rejected_tokens(self) -> int:
        return self.history_proposed_tokens - self.history_accepted_tokens


@dataclass(frozen=True, slots=True)
class HistoryPolicyResult:
    history_cap: int
    min_history_candidates: int
    min_generated_tokens: int
    opportunities: int
    opportunities_with_history: int
    actor_tokens: int
    semantic_candidates: int
    history_candidates: int
    proposals: int
    proposed_tokens: int
    accepted_tokens: int
    rejected_tokens: int
    acceptance_rate: float
    history_proposals: int
    history_proposed_tokens: int
    history_accepted_tokens: int
    history_rejected_tokens: int
    history_acceptance_rate: float
    target_steps: int
    target_steps_saved: int
    target_step_reduction: float
    incremental_target_steps_saved: int
    incremental_proposed_tokens: int
    incremental_accepted_tokens: int
    incremental_rejected_tokens: int
    per_opportunity_regressions: int


def _common_prefix(left: Sequence[int], right: Sequence[int]) -> int:
    matched = 0
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        matched += 1
    return matched


def simulate_serial_drafts(
    actual: Sequence[int],
    candidates: Sequence[RankedDraft],
    *,
    limit: int,
) -> Simulation:
    """Model one-shot serial drafts plus ordinary target-token steps.

    One verification step accepts the matching draft prefix and also samples
    the first target token after that prefix, matching ``BoundaryDraftStore``
    and the existing D3 tape analyzers.  Ineligible retrieval candidates stay
    pending until enough body tokens have been generated.
    """

    if limit <= 0:
        raise ValueError("limit must be positive")
    if any(candidate.min_generated_tokens < 0 for candidate in candidates):
        raise ValueError("min_generated_tokens must be non-negative")

    actual_tokens = tuple(int(token) for token in actual)
    fired = [False] * len(candidates)
    generated = target_steps = proposals = proposed = accepted = 0
    history_proposals = history_proposed = history_accepted = 0

    while generated < len(actual_tokens):
        offered = False
        for index, ranked in enumerate(candidates):
            if fired[index]:
                continue
            candidate = tuple(ranked.tokens[:limit])
            if (
                generated > len(candidate)
                or actual_tokens[:generated] != candidate[:generated]
            ):
                fired[index] = True
                continue
            if generated < ranked.min_generated_tokens:
                continue
            if generated == len(candidate):
                fired[index] = True
                continue

            proposal = candidate[generated:]
            matched = _common_prefix(proposal, actual_tokens[generated:])
            fired[index] = True
            offered = True
            target_steps += 1
            proposals += 1
            proposed += len(proposal)
            accepted += matched
            if ranked.source == "actor-history":
                history_proposals += 1
                history_proposed += len(proposal)
                history_accepted += matched
            generated += matched
            if generated < len(actual_tokens):
                generated += 1
            break

        if not offered:
            generated += 1
            target_steps += 1

    return Simulation(
        target_steps=target_steps,
        proposals=proposals,
        proposed_tokens=proposed,
        accepted_tokens=accepted,
        history_proposals=history_proposals,
        history_proposed_tokens=history_proposed,
        history_accepted_tokens=history_accepted,
    )


def _distinct(values: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    result: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for value in values:
        tokens = tuple(int(token) for token in value)
        if not tokens or tokens in seen:
            continue
        seen.add(tokens)
        result.append(tokens)
    return tuple(result)


def build_history_opportunities(
    tape: str,
    opportunities: Sequence[DraftOpportunity],
    *,
    known_primary_exact_sequences: Sequence[int] = (),
) -> tuple[HistoryOpportunity, ...]:
    """Attach only causally prior, recency-ordered Actor actions.

    ``known_primary_exact_sequences`` represents an externally observed
    semantic candidate already admitted by a prior ablation.  It is prepended
    to the semantic baseline and is never inferred from target tokens by this
    analyzer.
    """

    exact_sequences = {int(sequence) for sequence in known_primary_exact_sequences}
    history: list[tuple[int, ...]] = []
    result = []
    for opportunity in sorted(opportunities, key=lambda item: item.sequence):
        semantic = list(opportunity.candidate_tokens)
        if opportunity.sequence in exact_sequences:
            semantic.insert(0, opportunity.actual_tokens)
        result.append(
            HistoryOpportunity(
                tape=tape,
                sequence=opportunity.sequence,
                actual_tokens=opportunity.actual_tokens,
                semantic_candidates=_distinct(semantic),
                historical_candidates=tuple(history),
            )
        )
        actual = tuple(opportunity.actual_tokens)
        history = [actual, *(candidate for candidate in history if candidate != actual)]
    return tuple(result)


def ranked_policy_candidates(
    opportunity: HistoryOpportunity,
    *,
    history_cap: int,
    min_history_candidates: int,
    min_generated_tokens: int,
    max_candidates: int,
    limit: int,
) -> tuple[RankedDraft, ...]:
    if history_cap < 0:
        raise ValueError("history_cap must be non-negative")
    if min_generated_tokens < 0:
        raise ValueError("min_generated_tokens must be non-negative")
    if min_history_candidates <= 0:
        raise ValueError("min_history_candidates must be positive")
    if max_candidates <= 0:
        raise ValueError("max_candidates must be positive")
    if limit <= 0:
        raise ValueError("limit must be positive")

    semantic = opportunity.semantic_candidates[:max_candidates]
    ranked = [RankedDraft(tokens, "semantic") for tokens in semantic]
    seen = set(semantic)
    # The retention cap is applied before cross-source deduplication.  This
    # keeps both memory and the recency window truly bounded.
    retained = []
    for tokens in opportunity.historical_candidates[:history_cap]:
        if tokens in seen:
            continue
        seen.add(tokens)
        retained.append(tokens)
    available_slots = max_candidates - len(ranked)
    if len(retained) < min_history_candidates or available_slots < min_history_candidates:
        return tuple(ranked)
    retained = retained[:available_slots]
    if len(retained) < min_history_candidates:
        return tuple(ranked)

    truncated = [tuple(tokens[:limit]) for tokens in retained]
    for index, tokens in enumerate(retained):
        peers = truncated[:index] + truncated[index + 1 :]
        unique_prefix = 1 + max(
            (_common_prefix(truncated[index], peer) for peer in peers),
            default=0,
        )
        ranked.append(
            RankedDraft(
                tokens,
                "actor-history",
                min_generated_tokens=max(min_generated_tokens, unique_prefix),
            )
        )
    return tuple(ranked)


def _policy_metrics(
    opportunities: Sequence[HistoryOpportunity],
    *,
    history_cap: int,
    min_history_candidates: int,
    min_generated_tokens: int,
    max_candidates: int,
    limit: int,
) -> tuple[dict[str, int], tuple[Simulation, ...]]:
    simulations = []
    semantic_candidates = history_candidates = opportunities_with_history = 0
    for opportunity in opportunities:
        ranked = ranked_policy_candidates(
            opportunity,
            history_cap=history_cap,
            min_history_candidates=min_history_candidates,
            min_generated_tokens=min_generated_tokens,
            max_candidates=max_candidates,
            limit=limit,
        )
        semantic_candidates += sum(item.source == "semantic" for item in ranked)
        selected_history = sum(item.source == "actor-history" for item in ranked)
        history_candidates += selected_history
        opportunities_with_history += selected_history > 0
        simulations.append(
            simulate_serial_drafts(opportunity.actual_tokens, ranked, limit=limit)
        )
    totals = {
        "actor_tokens": sum(len(item.actual_tokens) for item in opportunities),
        "semantic_candidates": semantic_candidates,
        "history_candidates": history_candidates,
        "opportunities_with_history": opportunities_with_history,
        "target_steps": sum(item.target_steps for item in simulations),
        "proposals": sum(item.proposals for item in simulations),
        "proposed_tokens": sum(item.proposed_tokens for item in simulations),
        "accepted_tokens": sum(item.accepted_tokens for item in simulations),
        "history_proposals": sum(item.history_proposals for item in simulations),
        "history_proposed_tokens": sum(
            item.history_proposed_tokens for item in simulations
        ),
        "history_accepted_tokens": sum(
            item.history_accepted_tokens for item in simulations
        ),
    }
    return totals, tuple(simulations)


def analyze_history_caps(
    opportunities: Sequence[HistoryOpportunity],
    *,
    history_caps: Sequence[int],
    min_history_candidates: int = 3,
    min_generated_tokens: int = 1,
    max_candidates: int = 8,
    limit: int = 28,
) -> tuple[HistoryPolicyResult, ...]:
    baseline, baseline_simulations = _policy_metrics(
        opportunities,
        history_cap=0,
        min_history_candidates=min_history_candidates,
        min_generated_tokens=min_generated_tokens,
        max_candidates=max_candidates,
        limit=limit,
    )
    results = []
    for cap in history_caps:
        totals, simulations = _policy_metrics(
            opportunities,
            history_cap=cap,
            min_history_candidates=min_history_candidates,
            min_generated_tokens=min_generated_tokens,
            max_candidates=max_candidates,
            limit=limit,
        )
        proposed = totals["proposed_tokens"]
        accepted = totals["accepted_tokens"]
        history_proposed = totals["history_proposed_tokens"]
        history_accepted = totals["history_accepted_tokens"]
        actor_tokens = totals["actor_tokens"]
        target_steps = totals["target_steps"]
        results.append(
            HistoryPolicyResult(
                history_cap=cap,
                min_history_candidates=min_history_candidates,
                min_generated_tokens=min_generated_tokens,
                opportunities=len(opportunities),
                opportunities_with_history=totals["opportunities_with_history"],
                actor_tokens=actor_tokens,
                semantic_candidates=totals["semantic_candidates"],
                history_candidates=totals["history_candidates"],
                proposals=totals["proposals"],
                proposed_tokens=proposed,
                accepted_tokens=accepted,
                rejected_tokens=proposed - accepted,
                acceptance_rate=accepted / proposed if proposed else 0.0,
                history_proposals=totals["history_proposals"],
                history_proposed_tokens=history_proposed,
                history_accepted_tokens=history_accepted,
                history_rejected_tokens=history_proposed - history_accepted,
                history_acceptance_rate=(
                    history_accepted / history_proposed if history_proposed else 0.0
                ),
                target_steps=target_steps,
                target_steps_saved=actor_tokens - target_steps,
                target_step_reduction=(
                    (actor_tokens - target_steps) / actor_tokens
                    if actor_tokens
                    else 0.0
                ),
                incremental_target_steps_saved=(
                    baseline["target_steps"] - target_steps
                ),
                incremental_proposed_tokens=(
                    proposed - baseline["proposed_tokens"]
                ),
                incremental_accepted_tokens=(
                    accepted - baseline["accepted_tokens"]
                ),
                incremental_rejected_tokens=(
                    (proposed - accepted)
                    - (
                        baseline["proposed_tokens"]
                        - baseline["accepted_tokens"]
                    )
                ),
                per_opportunity_regressions=sum(
                    treatment.target_steps > control.target_steps
                    for control, treatment in zip(
                        baseline_simulations, simulations, strict=True
                    )
                ),
            )
        )
    return tuple(results)


def _exact_sequence_map(values: Sequence[str]) -> dict[str, tuple[int, ...]]:
    result: dict[str, list[int]] = {}
    for value in values:
        tape, separator, raw_sequences = value.partition("=")
        if not separator or not tape.strip():
            raise ValueError("known exact values must use TAPE=SEQUENCE[,SEQUENCE]")
        sequences = [
            int(item.strip())
            for item in raw_sequences.split(",")
            if item.strip()
        ]
        if not sequences:
            raise ValueError("known exact values must contain a sequence")
        result.setdefault(tape.strip(), []).extend(sequences)
    return {key: tuple(items) for key, items in result.items()}


def _per_tape(
    opportunities: Sequence[HistoryOpportunity],
    *,
    history_caps: Sequence[int],
    min_history_candidates: int,
    min_generated_tokens: int,
    max_candidates: int,
    limit: int,
) -> list[Mapping[str, Any]]:
    tapes = dict.fromkeys(item.tape for item in opportunities)
    return [
        {
            "tape": tape,
            "results": [
                asdict(result)
                for result in analyze_history_caps(
                    [item for item in opportunities if item.tape == tape],
                    history_caps=history_caps,
                    min_history_candidates=min_history_candidates,
                    min_generated_tokens=min_generated_tokens,
                    max_candidates=max_candidates,
                    limit=limit,
                )
            ],
        }
        for tape in tapes
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tape", type=Path, action="append", required=True)
    parser.add_argument("--actor-model", required=True)
    parser.add_argument("--drafter-model", required=True)
    parser.add_argument("--tokenizer", default="deepseek-ai/DeepSeek-V3")
    parser.add_argument("--revision")
    parser.add_argument("--format", default="tagged_json")
    parser.add_argument("--history-caps", default="0,1,2,3,4,8")
    parser.add_argument("--min-history-candidates", type=int, default=3)
    parser.add_argument("--min-generated-tokens", type=int, default=1)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--max-draft-tokens", type=int, default=28)
    parser.add_argument(
        "--known-primary-exact",
        action="append",
        default=[],
        metavar="TAPE=SEQUENCE[,SEQUENCE]",
    )
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        revision=args.revision,
    )
    exact = _exact_sequence_map(args.known_primary_exact)
    all_opportunities = []
    for tape in args.tape:
        drafted = build_opportunities(
            parse_tape(tape, args.format),
            actor_model=args.actor_model,
            drafter_model=args.drafter_model,
            tokenizer=tokenizer,
            drafter_width=2,
            drafter_completion_limit=1,
        )
        all_opportunities.extend(
            build_history_opportunities(
                tape.name,
                drafted,
                known_primary_exact_sequences=exact.get(tape.name, ()),
            )
        )

    caps = tuple(
        int(item.strip())
        for item in args.history_caps.split(",")
        if item.strip()
    )
    results = analyze_history_caps(
        all_opportunities,
        history_caps=caps,
        min_history_candidates=args.min_history_candidates,
        min_generated_tokens=args.min_generated_tokens,
        max_candidates=args.max_candidates,
        limit=args.max_draft_tokens,
    )
    print(
        json.dumps(
            {
                "tokenizer": args.tokenizer,
                "tokenizer_revision": (
                    args.revision or tokenizer.init_kwargs.get("_commit_hash")
                ),
                "format": args.format,
                "policy": {
                    "drafter_width": 2,
                    "drafter_completion_limit": 1,
                    "semantic_candidates_first": True,
                    "history_recency": "most-recent-distinct-first",
                    "history_activation": "shortest-unique-prefix",
                    "min_history_candidates": args.min_history_candidates,
                    "min_generated_tokens": args.min_generated_tokens,
                    "max_candidates": args.max_candidates,
                    "max_draft_tokens": args.max_draft_tokens,
                    "known_primary_exact": exact,
                },
                "pooled": [asdict(result) for result in results],
                "per_tape": _per_tape(
                    all_opportunities,
                    history_caps=caps,
                    min_history_candidates=args.min_history_candidates,
                    min_generated_tokens=args.min_generated_tokens,
                    max_candidates=args.max_candidates,
                    limit=args.max_draft_tokens,
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
