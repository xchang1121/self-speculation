"""Measure agreement-cap drafting on a real Transformers target verifier.

The target output is generated once and then held fixed. Candidate cohorts
cover exact-first, exact-second, and no-exact bundles in the 3:3:6 proportions
observed in the strict width-two tape replay. Both policies use the same target
model, output, candidate bodies, and generation settings.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from self_speculation import InferenceRequest, TransformersEngine
from self_speculation.drafts.store import DraftProposal

if __package__:
    from .d3_agreement_cap_ablation import _shared_prefix_length
    from .transformers_d3_ablation import (
        DEFAULT_MODEL_REVISION,
        RunMeasurement,
        _measure,
    )
else:
    from d3_agreement_cap_ablation import (  # type: ignore[no-redef]
        _shared_prefix_length,
    )
    from transformers_d3_ablation import (  # type: ignore[no-redef]
        DEFAULT_MODEL_REVISION,
        RunMeasurement,
        _measure,
    )


@dataclass(slots=True)
class AgreementReplayStore:
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
        try:
            boundary_index = max(
                index
                for index, token in enumerate(generated)
                if token == self.boundary_token_id
            )
        except ValueError:
            return None
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


@dataclass(frozen=True, slots=True)
class PolicyMeasurement:
    policy: str
    scenario: str
    elapsed_ms: float
    forward_calls: int
    proposed_tokens: int
    accepted_tokens: int
    rejected_tokens: int
    proposals: int
    clipped_proposals: int
    clipped_tokens: int


def _different_token(token: int, *, vocabulary_size: int, offset: int) -> int:
    selected = (token + offset) % vocabulary_size
    return selected if selected != token else (selected + 1) % vocabulary_size


def _wrong_candidate(
    actual: Sequence[int],
    *,
    shared_prefix_tokens: int,
    vocabulary_size: int,
    offset: int,
) -> tuple[int, ...]:
    candidate = list(actual)
    candidate[shared_prefix_tokens] = _different_token(
        candidate[shared_prefix_tokens],
        vocabulary_size=vocabulary_size,
        offset=offset,
    )
    return tuple(candidate)


async def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
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
    request = InferenceRequest(prompt=args.prompt, max_tokens=args.max_new_tokens)
    prompt_tokens = await baseline_engine.prompt_token_count(request)
    reference = await _measure(baseline_engine, model, request)
    if len(reference.output_token_ids) <= args.shared_prefix_tokens + 2:
        raise RuntimeError("reference output is too short for the requested branch")
    boundary = reference.output_token_ids[0]
    actual = tuple(
        reference.output_token_ids[
            1 : 1 + min(args.max_draft_tokens, len(reference.output_token_ids) - 1)
        ]
    )
    first_wrong = _wrong_candidate(
        actual,
        shared_prefix_tokens=args.shared_prefix_tokens,
        vocabulary_size=int(model.config.vocab_size),
        offset=1,
    )
    second_wrong = _wrong_candidate(
        actual,
        shared_prefix_tokens=args.shared_prefix_tokens,
        vocabulary_size=int(model.config.vocab_size),
        offset=2,
    )
    scenarios = {
        "exact-first": (actual, first_wrong),
        "exact-second": (first_wrong, actual),
        "no-exact": (first_wrong, second_wrong),
    }
    weights = {"exact-first": 3, "exact-second": 3, "no-exact": 6}

    async def measure(policy: str, scenario: str) -> PolicyMeasurement:
        store = AgreementReplayStore(
            candidates=scenarios[scenario],
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
        result = await _measure(
            engine,
            model,
            InferenceRequest(
                prompt=args.prompt,
                request_id=f"{policy}-{scenario}",
                max_tokens=args.max_new_tokens,
            ),
        )
        if result.output_token_ids != reference.output_token_ids:
            raise RuntimeError(f"{policy}/{scenario} changed target output")
        return PolicyMeasurement(
            policy=policy,
            scenario=scenario,
            elapsed_ms=result.elapsed_ms,
            forward_calls=result.forward_calls,
            proposed_tokens=store.proposed_tokens,
            accepted_tokens=store.accepted_tokens,
            rejected_tokens=store.proposed_tokens - store.accepted_tokens,
            proposals=store.proposals,
            clipped_proposals=store.clipped_proposals,
            clipped_tokens=store.clipped_tokens,
        )

    # Warm every distinct path before measuring alternating policy order.
    for scenario in scenarios:
        for policy in ("full", "agreement-cap"):
            await measure(policy, scenario)

    measurements: list[PolicyMeasurement] = []
    for repeat in range(args.repeats):
        policies = (
            ("full", "agreement-cap")
            if repeat % 2 == 0
            else ("agreement-cap", "full")
        )
        for scenario in scenarios:
            for policy in policies:
                measurements.append(await measure(policy, scenario))

    per_scenario: dict[str, dict[str, Any]] = {}
    for scenario in scenarios:
        per_scenario[scenario] = {}
        for policy in ("full", "agreement-cap"):
            selected = [
                value
                for value in measurements
                if value.scenario == scenario and value.policy == policy
            ]
            per_scenario[scenario][policy] = {
                "median_elapsed_ms": statistics.median(
                    value.elapsed_ms for value in selected
                ),
                "median_forward_calls": statistics.median(
                    value.forward_calls for value in selected
                ),
                "proposed_tokens": selected[0].proposed_tokens,
                "accepted_tokens": selected[0].accepted_tokens,
                "rejected_tokens": selected[0].rejected_tokens,
                "proposals": selected[0].proposals,
                "clipped_proposals": selected[0].clipped_proposals,
                "clipped_tokens": selected[0].clipped_tokens,
            }

    pooled: dict[str, Any] = {}
    for policy in ("full", "agreement-cap"):
        # Per-scenario medians are less sensitive to execution order; apply the
        # observed tape weights only after each distinct path is summarized.
        pooled[policy] = {
            "weighted_sum_median_elapsed_ms": sum(
                weights[scenario]
                * per_scenario[scenario][policy]["median_elapsed_ms"]
                for scenario in scenarios
            ),
            **{
                f"weighted_{metric}": sum(
                    weights[scenario] * per_scenario[scenario][policy][metric]
                    for scenario in scenarios
                )
                for metric in (
                    "median_forward_calls",
                    "proposed_tokens",
                    "accepted_tokens",
                    "rejected_tokens",
                    "proposals",
                    "clipped_proposals",
                    "clipped_tokens",
                )
            },
        }

    full_ms = pooled["full"]["weighted_sum_median_elapsed_ms"]
    clipped_ms = pooled["agreement-cap"]["weighted_sum_median_elapsed_ms"]
    return {
        "model": args.model,
        "revision": args.revision
        or tokenizer.init_kwargs.get("_commit_hash")
        or getattr(model.config, "_commit_hash", None),
        "device": str(model.device),
        "torch": torch.__version__,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": len(reference.output_token_ids),
        "draft_tokens": len(actual),
        "shared_prefix_tokens": args.shared_prefix_tokens,
        "repeats": args.repeats,
        "scenario_weights": weights,
        "outputs_identical": True,
        "per_scenario": per_scenario,
        "pooled_weighted": pooled,
        "agreement_wall_time_ratio": clipped_ms / full_ms if full_ms else 0.0,
        "measurements": [asdict(value) for value in measurements],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="hf-internal-testing/tiny-random-gpt2",
    )
    parser.add_argument("--revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--prompt", default="Fix the function and run its tests:")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-draft-tokens", type=int, default=28)
    parser.add_argument("--shared-prefix-tokens", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.max_new_tokens <= 2:
        parser.error("--max-new-tokens must be greater than two")
    if args.max_draft_tokens <= 1:
        parser.error("--max-draft-tokens must be greater than one")
    if args.shared_prefix_tokens <= 0:
        parser.error("--shared-prefix-tokens must be positive")
    if args.shared_prefix_tokens >= args.max_draft_tokens:
        parser.error("--shared-prefix-tokens must be below --max-draft-tokens")
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    print(json.dumps(asyncio.run(run(args)), indent=2))


if __name__ == "__main__":
    main()
