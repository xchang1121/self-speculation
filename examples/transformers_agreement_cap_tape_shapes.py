"""Replay strict tape candidate shapes through a real Transformers verifier.

DeepSeek token equality, candidate lengths, ordering, and exact/miss positions
are mapped position-wise onto one deterministic tiny-GPT2 continuation. This
preserves the verifier control flow while making target forwards and CPU wall
time directly measurable without exposing tape prompts to the model.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from self_speculation import InferenceRequest, TransformersEngine
from self_speculation.drafts.store import DraftProposal

if __package__:
    from .d3_agreement_cap_ablation import _shared_prefix_length
    from .d3_tape_ablation import DraftOpportunity, build_opportunities, parse_tape
    from .transformers_d3_ablation import (
        DEFAULT_MODEL_REVISION,
        _measure,
    )
else:
    from d3_agreement_cap_ablation import (  # type: ignore[no-redef]
        _shared_prefix_length,
    )
    from d3_tape_ablation import (  # type: ignore[no-redef]
        DraftOpportunity,
        build_opportunities,
        parse_tape,
    )
    from transformers_d3_ablation import (  # type: ignore[no-redef]
        DEFAULT_MODEL_REVISION,
        _measure,
    )


@dataclass(frozen=True, slots=True)
class TapeShapeMeasurement:
    opportunity: int
    policy: str
    elapsed_ms: float
    forward_calls: int
    proposed_tokens: int
    accepted_tokens: int
    rejected_tokens: int
    proposals: int
    clipped_proposals: int
    clipped_tokens: int


@dataclass(slots=True)
class AgreementReplayStore:
    """Minimal store stub for the two proposal-length policies under test."""

    candidates: tuple[tuple[int, ...], ...]
    prompt_token_count: int
    boundary_token_id: int
    policy: str
    max_draft_tokens: int
    proposed_tokens: int = 0
    accepted_tokens: int = 0
    proposals: int = 0
    clipped_proposals: int = 0
    clipped_tokens: int = 0
    _active: list[tuple[int, ...]] = field(init=False, repr=False)
    _last_offer_sequence_length: int | None = field(default=None, init=False)
    _last_proposal_length: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.policy not in {"full", "agreement-cap"}:
            raise ValueError(f"unsupported agreement replay policy: {self.policy}")
        if self.max_draft_tokens <= 0:
            raise ValueError("max_draft_tokens must be positive")
        self._active = [
            tuple(candidate[: self.max_draft_tokens])
            for candidate in self.candidates
            if candidate
        ]

    def offer(
        self,
        request_id: str,
        sequence_token_ids: Sequence[int],
        *,
        sequence_length: int | None = None,
        max_tokens: int | None = None,
    ) -> DraftProposal | None:
        if sequence_length is None:
            sequence_length = len(sequence_token_ids)
        if self._last_offer_sequence_length == sequence_length:
            return None
        generated = tuple(
            int(token)
            for token in sequence_token_ids[
                min(self.prompt_token_count, sequence_length) : sequence_length
            ]
        )
        boundary_indices = [
            index
            for index, token in enumerate(generated)
            if token == self.boundary_token_id
        ]
        if not boundary_indices:
            return None
        boundary_index = boundary_indices[-1]
        generated_body = generated[boundary_index + 1 :]
        self._active = [
            candidate
            for candidate in self._active
            if len(generated_body) < len(candidate)
            and generated_body == candidate[: len(generated_body)]
        ]
        if not self._active:
            return None

        primary = self._active[0]
        primary_suffix = primary[len(generated_body) :]
        proposal = primary_suffix
        clipped = False
        if self.policy == "agreement-cap" and len(self._active) > 1:
            suffixes = [
                candidate[len(generated_body) :]
                for candidate in self._active
            ]
            shared = _shared_prefix_length(suffixes)
            if shared > 0 and shared < len(primary_suffix):
                proposal = primary_suffix[:shared]
                clipped = True

        limit = min(max_tokens or self.max_draft_tokens, self.max_draft_tokens)
        proposal = proposal[:limit]
        if not proposal:
            return None
        if clipped:
            self.clipped_proposals += 1
            self.clipped_tokens += len(primary_suffix) - len(proposal)
        else:
            self._active.pop(0)
        self._last_offer_sequence_length = sequence_length
        self._last_proposal_length = len(proposal)
        self.proposals += 1
        self.proposed_tokens += len(proposal)
        return DraftProposal(
            request_id=request_id,
            token_ids=tuple(proposal),
            skipped_prefix_tokens=len(generated_body),
            generated_body_tokens=len(generated_body),
            boundary_index=boundary_index,
            candidate_count=len(self.candidates),
        )

    def observe_acceptance(self, request_id: str, accepted_tokens: int) -> bool:
        del request_id
        if not self._last_proposal_length:
            return False
        accepted = max(0, min(int(accepted_tokens), self._last_proposal_length))
        self.accepted_tokens += accepted
        self._last_proposal_length = 0
        return True

    def take_outcome(self, request_id: str) -> None:
        del request_id
        return None


def remap_candidate_shapes(
    opportunity: DraftOpportunity,
    target_body: Sequence[int],
    *,
    vocabulary_size: int,
    limit: int,
) -> tuple[tuple[int, ...], ...]:
    """Preserve same-position equality relations on a target-model sequence."""

    if vocabulary_size <= 1:
        raise ValueError("vocabulary_size must exceed one")
    if limit <= 0:
        raise ValueError("limit must be positive")
    mapped_by_position: list[dict[int, int]] = []
    used_by_position: list[set[int]] = []
    candidates = []
    for source_candidate in opportunity.candidate_tokens:
        mapped = []
        for position, source_token in enumerate(source_candidate[:limit]):
            if position >= len(target_body):
                raise ValueError("target body is shorter than a mapped candidate")
            while len(mapped_by_position) <= position:
                mapped_by_position.append({})
                used_by_position.append({int(target_body[len(used_by_position)])})
            if (
                position < len(opportunity.actual_tokens)
                and source_token == opportunity.actual_tokens[position]
            ):
                target_token = int(target_body[position])
            else:
                position_map = mapped_by_position[position]
                target_token = position_map.get(int(source_token), -1)
                if target_token < 0:
                    selected = (int(source_token) + position + 1) % vocabulary_size
                    while selected in used_by_position[position]:
                        selected = (selected + 1) % vocabulary_size
                    position_map[int(source_token)] = selected
                    used_by_position[position].add(selected)
                    target_token = selected
            mapped.append(target_token)
        candidates.append(tuple(mapped))
    return tuple(candidates)


async def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    source_tokenizer = AutoTokenizer.from_pretrained(
        args.source_tokenizer,
        revision=args.source_revision,
    )
    opportunities = tuple(
        opportunity
        for tape in args.tape
        for opportunity in build_opportunities(
            parse_tape(tape, args.format),
            actor_model=args.actor_model,
            drafter_model=args.drafter_model,
            tokenizer=source_tokenizer,
            drafter_width=args.drafter_width,
            drafter_completion_limit=args.drafter_completion_limit,
        )
    )
    if not opportunities:
        raise RuntimeError("the tapes produced no strict action opportunities")

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
    ).eval()
    generation_kwargs = {
        "do_sample": False,
        "eos_token_id": None,
        "pad_token_id": tokenizer.eos_token_id or 0,
    }
    baseline_engine = TransformersEngine(
        model,
        tokenizer,
        generation_kwargs=generation_kwargs,
    )
    prompt_tokens = await baseline_engine.prompt_token_count(
        InferenceRequest(prompt=args.prompt)
    )
    max_body_tokens = max(len(value.actual_tokens) for value in opportunities)
    reference = await _measure(
        baseline_engine,
        model,
        InferenceRequest(
            prompt=args.prompt,
            max_tokens=max_body_tokens + 2,
        ),
    )
    if len(reference.output_token_ids) < max_body_tokens + 2:
        raise RuntimeError("target reference ended before the longest tape shape")
    boundary = reference.output_token_ids[0]
    target_body = reference.output_token_ids[1:]
    mapped_candidates = tuple(
        remap_candidate_shapes(
            opportunity,
            target_body,
            vocabulary_size=int(model.config.vocab_size),
            limit=args.max_draft_tokens,
        )
        for opportunity in opportunities
    )

    async def measure(opportunity_index: int, policy: str) -> TapeShapeMeasurement:
        opportunity = opportunities[opportunity_index]
        store = AgreementReplayStore(
            candidates=mapped_candidates[opportunity_index],
            prompt_token_count=prompt_tokens,
            boundary_token_id=boundary,
            policy=policy,
            max_draft_tokens=args.max_draft_tokens,
        )
        engine = TransformersEngine(
            model,
            tokenizer,
            generation_kwargs=generation_kwargs,
            draft_store=store,  # type: ignore[arg-type]
            max_draft_tokens=args.max_draft_tokens,
        )
        # Keep one authoritative token after the mapped action body. Without
        # it, max-length termination can leave the final draft token unverified
        # and make Transformers' num_matches incomparable with the tape model.
        expected = reference.output_token_ids[: len(opportunity.actual_tokens) + 2]
        result = await _measure(
            engine,
            model,
            InferenceRequest(
                prompt=args.prompt,
                request_id=f"shape-{opportunity_index}-{policy}",
                max_tokens=len(expected),
            ),
        )
        if result.output_token_ids != expected:
            raise RuntimeError(
                f"{policy}/opportunity-{opportunity_index} changed target output"
            )
        return TapeShapeMeasurement(
            opportunity=opportunity_index,
            policy=policy,
            elapsed_ms=result.elapsed_ms,
            forward_calls=result.forward_calls,
            proposed_tokens=store.proposed_tokens,
            accepted_tokens=store.accepted_tokens,
            rejected_tokens=store.proposed_tokens - store.accepted_tokens,
            proposals=store.proposals,
            clipped_proposals=store.clipped_proposals,
            clipped_tokens=store.clipped_tokens,
        )

    for opportunity_index in range(len(opportunities)):
        for policy in ("full", "agreement-cap"):
            await measure(opportunity_index, policy)

    measurements = []
    for repeat in range(args.repeats):
        policies = (
            ("full", "agreement-cap")
            if repeat % 2 == 0
            else ("agreement-cap", "full")
        )
        for opportunity_index in range(len(opportunities)):
            for policy in policies:
                measurements.append(await measure(opportunity_index, policy))

    summarized = []
    for opportunity_index in range(len(opportunities)):
        for policy in ("full", "agreement-cap"):
            selected = [
                value
                for value in measurements
                if value.opportunity == opportunity_index and value.policy == policy
            ]
            summarized.append(
                TapeShapeMeasurement(
                    opportunity=opportunity_index,
                    policy=policy,
                    elapsed_ms=statistics.median(
                        value.elapsed_ms for value in selected
                    ),
                    forward_calls=int(
                        statistics.median(value.forward_calls for value in selected)
                    ),
                    proposed_tokens=selected[0].proposed_tokens,
                    accepted_tokens=selected[0].accepted_tokens,
                    rejected_tokens=selected[0].rejected_tokens,
                    proposals=selected[0].proposals,
                    clipped_proposals=selected[0].clipped_proposals,
                    clipped_tokens=selected[0].clipped_tokens,
                )
            )

    pooled = {}
    for policy in ("full", "agreement-cap"):
        selected = [value for value in summarized if value.policy == policy]
        pooled[policy] = {
            "sum_median_elapsed_ms": sum(value.elapsed_ms for value in selected),
            **{
                field: sum(getattr(value, field) for value in selected)
                for field in (
                    "forward_calls",
                    "proposed_tokens",
                    "accepted_tokens",
                    "rejected_tokens",
                    "proposals",
                    "clipped_proposals",
                    "clipped_tokens",
                )
            },
        }
    full_ms = pooled["full"]["sum_median_elapsed_ms"]
    agreement_ms = pooled["agreement-cap"]["sum_median_elapsed_ms"]
    return {
        "target_model": args.model,
        "target_revision": args.revision
        or tokenizer.init_kwargs.get("_commit_hash")
        or getattr(model.config, "_commit_hash", None),
        "source_tokenizer": args.source_tokenizer,
        "source_revision": args.source_revision
        or source_tokenizer.init_kwargs.get("_commit_hash"),
        "device": str(model.device),
        "torch": torch.__version__,
        "tapes": len(args.tape),
        "opportunities": len(opportunities),
        "drafter_width": args.drafter_width,
        "drafter_completion_limit": args.drafter_completion_limit,
        "max_draft_tokens": args.max_draft_tokens,
        "repeats": args.repeats,
        "outputs_identical": True,
        "pooled": pooled,
        "agreement_wall_time_ratio": agreement_ms / full_ms if full_ms else 0.0,
        "opportunity_medians": [asdict(value) for value in summarized],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tape", type=Path, action="append", required=True)
    parser.add_argument("--actor-model", required=True)
    parser.add_argument("--drafter-model", required=True)
    parser.add_argument("--source-tokenizer", default="deepseek-ai/DeepSeek-V3")
    parser.add_argument("--source-revision")
    parser.add_argument("--format", default="tagged_json")
    parser.add_argument("--drafter-width", type=int, default=2)
    parser.add_argument("--drafter-completion-limit", type=int)
    parser.add_argument("--max-draft-tokens", type=int, default=28)
    parser.add_argument(
        "--model",
        default="hf-internal-testing/tiny-random-gpt2",
    )
    parser.add_argument("--revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--prompt", default="Fix the function and run its tests:")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.drafter_width <= 0:
        parser.error("--drafter-width must be positive")
    if (
        args.drafter_completion_limit is not None
        and args.drafter_completion_limit <= 0
    ):
        parser.error("--drafter-completion-limit must be positive")
    if args.max_draft_tokens <= 0:
        parser.error("--max-draft-tokens must be positive")
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    print(json.dumps(asyncio.run(run(args)), indent=2))


if __name__ == "__main__":
    main()
