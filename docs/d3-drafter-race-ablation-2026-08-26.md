# D3 Drafter hedge-cancellation ablation — 2026-08-26

## Decision

The target-verifier gate passes: after selecting the retained dispatch width of
two, keeping only the first completed response containing a valid tool call does
not change any D3 proposal, accepted/rejected token count, target step, target
forward, or target output on the three useful recorded tapes.

This result permits a Pi-side product experiment using its existing per-request
`AbortSignal`. It does not by itself claim wall-clock speedup; the companion Pi
action analyzer measures only 132.650 ms (7.78%) of residual Drafter service as
counterfactually removable.

## Motivation and boundary

Standard request hedging accepts a successful response and cancels outstanding
attempts. Drafter samples are stochastic, not equivalent replicas, so action
exactness alone is insufficient: a late wrong full action can still share a
useful prefix with the Actor and become a serial D3 fallback candidate.

The treatment is therefore strictly defined as:

1. Select the first two same-context requests in dispatch order, matching the
   retained Pi default.
2. Sort only those selected requests by recorded completion duration.
3. Ignore empty/malformed responses when choosing a winner.
4. Retain candidates from the first valid response only.
5. Use the current `tagged_json` formatter, DeepSeek V3 tokenizer, and D3 cap
   28 without changing candidate tokens or verifier semantics.

Primary lifecycle references:

- Dean and Barroso, *The Tail at Scale*:
  https://research.google/pubs/the-tail-at-scale/
- gRPC Request Hedging:
  https://grpc.io/docs/guides/request-hedging/
- gRPC Cancellation:
  https://grpc.io/docs/guides/cancellation/

## Strict token replay

Environment and inputs:

- tapes: `deepseek-mock-deterministic.json`, `deepseek-mock-success.json`, and
  `pattern-learning.json`
- source tokenizer: `deepseek-ai/DeepSeek-V3`, revision
  `e815299b0bcbac849fa540c768ef21845365c9eb`
- formatter: `tagged_json`
- Drafter dispatch width: 2
- maximum draft tokens: 28

`deepseek-live.json` remains excluded because its responses are
insufficient-balance errors rather than model output.

| Tape | Candidates full → first-valid | Proposals | Proposed / accepted / rejected | Target steps | Steps saved |
| --- | ---: | ---: | ---: | ---: | ---: |
| deterministic mock | 4 → 3 | 3 → 3 | 56 / 46 / 10 → same | 63 → 63 | 44 → 44 |
| success mock | 4 → 3 | 3 → 3 | 44 / 34 / 10 → same | 63 → 63 | 32 → 32 |
| pattern learning | 10 → 6 | 6 → 6 | 98 / 64 / 34 → same | 152 → 152 | 62 → 62 |
| **Pooled** | **18 → 12** | **12 → 12** | **198 / 144 / 54 → same** | **278 → 278** | **138 → 138** |

Six late candidate bodies disappear, but none is reached by the current serial
verifier. Every opportunity's first-valid candidate produces the same accepted
prefix and authoritative bonus-token path as full width two.

## Pinned real Transformers verifier

The existing tape-shape harness now accepts a generic
`--drafter-completion-limit`. Separate full-width and limit-one runs use:

- target: `hf-internal-testing/tiny-random-gpt2`, revision
  `71034c5d8bde858ff824298bdedc65515b97d2b9`
- PyTorch `2.13.0+cpu`, Transformers `5.15.1`
- 12 opportunities and 11 warm-after-path alternating-policy repetitions

| Candidate admission | Output | Target forwards | Proposals | Proposed / accepted / rejected |
| --- | ---: | ---: | ---: | ---: |
| Full width two | identical | 296 | 12 | 198 / 144 / 54 |
| First valid only | identical | 296 | 12 | 198 / 144 / 54 |

The two invocations' wall-time medians are intentionally not compared: they ran
in separate processes and showed large CPU timing drift. Forward and token
counts are deterministic and reproduce the offline replay exactly.

## Implementation and reproduction

`build_opportunities` now separates two causal axes:

- `drafter_width` truncates in dispatch order;
- `drafter_completion_limit` truncates valid responses after completion order.

The generic `analyze_opportunities` helper measures an already-selected policy,
preventing the new experiment from duplicating verifier simulation logic.

Run the offline comparison:

```powershell
.\.venv\Scripts\python.exe examples\d3_drafter_race_ablation.py `
  --tape <deepseek-mock-deterministic.json> `
  --tape <deepseek-mock-success.json> `
  --tape <pattern-learning.json> `
  --actor-model deepseek-v4-pro `
  --drafter-model deepseek-v4-flash `
  --revision e815299b0bcbac849fa540c768ef21845365c9eb `
  --drafter-width 2 --max-draft-tokens 28
```

Add `--drafter-completion-limit 1` to the pinned real-model shape replay for
the treatment. No engine, protocol, target store, or product default changes in
this Python ablation commit.

## Limitations

- The evidence contains 12 opportunities and related mock recordings. New
  models or broader candidate distributions can make a late fallback useful.
- The verifier is serial. Parallel tree verifiers may benefit from candidate
  diversity even when this store does not reach a second branch.
- Target-side work is unchanged, not reduced. The expected gain is solely the
  Pi/provider residual generation avoided after the first valid completion.
- Provider cancellation must be honored and must remain distinguishable from a
  real Drafter error in Pi telemetry.
