"""Run a real Transformers target-verifier A/B with an exact replay draft.

The exact draft isolates D3 integration overhead from prediction quality. It
measures target forward calls and wall time but does not estimate live-model hit
rate: the candidate is intentionally copied from a baseline generation.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Iterator

from self_speculation import (
    BoundaryDraftStore,
    DraftBoundary,
    DraftRequest,
    InferenceRequest,
    TransformersEngine,
)


DEFAULT_MODEL_REVISION = "71034c5d8bde858ff824298bdedc65515b97d2b9"


@dataclass(frozen=True, slots=True)
class RunMeasurement:
    elapsed_ms: float
    forward_calls: int
    output_token_ids: tuple[int, ...]


@contextmanager
def _count_forwards(model: Any) -> Iterator[list[int]]:
    calls = [0]

    def count(*args: Any) -> None:
        del args
        calls[0] += 1

    handle = model.register_forward_hook(count)
    try:
        yield calls
    finally:
        handle.remove()


async def _measure(
    engine: TransformersEngine,
    model: Any,
    request: InferenceRequest,
) -> RunMeasurement:
    started = time.perf_counter()
    with _count_forwards(model) as calls:
        chunks = [chunk async for chunk in engine.stream(request)]
    return RunMeasurement(
        elapsed_ms=(time.perf_counter() - started) * 1000,
        forward_calls=calls[0],
        output_token_ids=tuple(
            token_id for chunk in chunks for token_id in chunk.token_ids
        ),
    )


def _median(values: list[float | int]) -> float:
    return float(statistics.median(values))


def _run_summary(measurement: RunMeasurement) -> dict[str, float | int]:
    return {
        "elapsed_ms": measurement.elapsed_ms,
        "forward_calls": measurement.forward_calls,
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
    )
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

    reference = await _measure(
        baseline_engine,
        model,
        InferenceRequest(prompt=args.prompt, max_tokens=args.max_new_tokens),
    )
    if len(reference.output_token_ids) < 2:
        raise RuntimeError("reference generation did not produce enough tokens")
    draft_tokens = reference.output_token_ids[
        1 : 1 + min(args.max_draft_tokens, len(reference.output_token_ids) - 1)
    ]

    store = BoundaryDraftStore(max_draft_tokens=args.max_draft_tokens)
    assisted_engine = TransformersEngine(
        model,
        tokenizer,
        generation_kwargs=generation_kwargs,
        draft_store=store,
        max_draft_tokens=args.max_draft_tokens,
    )

    async def assisted(run_index: int, *, warmup: bool = False) -> RunMeasurement:
        request_id = f"transformers-d3-{'warmup' if warmup else run_index}"
        store.register(
            DraftRequest(
                request_id=request_id,
                token_ids=draft_tokens,
                boundary=DraftBoundary(token_ids=(reference.output_token_ids[0],)),
                prompt_token_count=prompt_tokens,
            )
        )
        result = await _measure(
            assisted_engine,
            model,
            InferenceRequest(
                prompt=args.prompt,
                request_id=request_id,
                max_tokens=args.max_new_tokens,
            ),
        )
        store.clear(request_id)
        return result

    # Warm both generation modes before collecting wall-clock samples.
    await _measure(
        baseline_engine,
        model,
        InferenceRequest(prompt=args.prompt, max_tokens=args.max_new_tokens),
    )
    warm_assisted = await assisted(-1, warmup=True)
    if warm_assisted.output_token_ids != reference.output_token_ids:
        raise RuntimeError("assisted warm-up changed target output")

    baselines: list[RunMeasurement] = []
    assisted_runs: list[RunMeasurement] = []
    for run_index in range(args.repeats):
        baseline = await _measure(
            baseline_engine,
            model,
            InferenceRequest(prompt=args.prompt, max_tokens=args.max_new_tokens),
        )
        accelerated = await assisted(run_index)
        if baseline.output_token_ids != reference.output_token_ids:
            raise RuntimeError("baseline generation was not deterministic")
        if accelerated.output_token_ids != reference.output_token_ids:
            raise RuntimeError("assisted decoding changed target output")
        baselines.append(baseline)
        assisted_runs.append(accelerated)

    baseline_ms = _median([item.elapsed_ms for item in baselines])
    assisted_ms = _median([item.elapsed_ms for item in assisted_runs])
    baseline_forwards = _median([item.forward_calls for item in baselines])
    assisted_forwards = _median([item.forward_calls for item in assisted_runs])
    return {
        "model": args.model,
        "revision": args.revision
        or tokenizer.init_kwargs.get("_commit_hash")
        or getattr(model.config, "_commit_hash", None),
        "device": str(model.device),
        "torch": torch.__version__,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": len(reference.output_token_ids),
        "draft_tokens": len(draft_tokens),
        "repeats": args.repeats,
        "outputs_identical": True,
        "baseline": {
            "median_elapsed_ms": baseline_ms,
            "median_forward_calls": baseline_forwards,
            "runs": [_run_summary(item) for item in baselines],
        },
        "assisted": {
            "median_elapsed_ms": assisted_ms,
            "median_forward_calls": assisted_forwards,
            "runs": [_run_summary(item) for item in assisted_runs],
        },
        "forward_call_reduction": (
            (baseline_forwards - assisted_forwards) / baseline_forwards
            if baseline_forwards
            else 0.0
        ),
        "wall_time_speedup": baseline_ms / assisted_ms if assisted_ms else 0.0,
        "store": asdict(store.snapshot()),
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
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.max_new_tokens <= 1:
        parser.error("--max-new-tokens must be greater than one")
    if args.max_draft_tokens <= 0:
        parser.error("--max-draft-tokens must be positive")
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    print(json.dumps(asyncio.run(run(args)), indent=2))


if __name__ == "__main__":
    main()
