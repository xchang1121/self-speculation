# Source-neutral candidate-ranking ablation — 2026-08-25

This experiment asks whether target-token structure can improve the ordering of
otherwise tied action candidates without preferring Drafter or PatternAware by
source name.

## Candidate and safety boundary

Tree verifiers such as [SpecInfer](https://arxiv.org/abs/2305.09781) and
[Multi-Candidate Speculative Decoding](https://arxiv.org/abs/2401.06706) verify
multiple branches in parallel. [EAGLE-2](https://arxiv.org/abs/2406.16858)
instead uses draft confidence to shape a dynamic tree. The current portable
`BoundaryDraftStore` exposes one linear
candidate per verifier round, and ordinary Drafter action candidates carry no
comparable confidence. Pretending that serial fallback is a parallel tree, or
inventing a source prior, would therefore overstate both evidence and speedup.

The bounded candidate tested here is a prefix-consensus medoid. Within an
equal-confidence group, it ranks each candidate by the sum of target-token
common-prefix lengths with every candidate in the group (including itself),
then keeps original completion order for an exact tie. Under a uniform prior
over tied candidates this maximizes immediate expected prefix agreement. The
rule never reads source labels and never crosses an existing score boundary.

## Replay method

`examples/d3_tape_ablation.py` now compares recorded completion order with
`prefix-consensus`. It decodes complete Drafter/Actor tool calls from the three
useful private tapes, uses the current `tagged_json` formatter, and tokenizes
with `deepseek-ai/DeepSeek-V3` revision
`e815299b0bcbac849fa540c768ef21845365c9eb`. K is fixed at the accepted D3 cap
of 28. As in the D3 experiment, target-step counts are a verifier-work proxy,
not a wall-clock claim.

## Pooled result

| Ordering | Candidates | Proposed | Accepted | Rejected | Acceptance | Target steps saved |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Recorded completion | 21 | 198 | 144 | 54 | 72.7% | 138 |
| Prefix consensus | 21 | 207 | 144 | 63 | 69.6% | 138 |

The deterministic and mock-success tapes were unchanged. On
`pattern-learning.json`, prefix consensus changed an early choice but recovered
no additional accepted token or target step: proposed work rose from 98 to 107
tokens and rejected work from 34 to 43. Pooled rejected verification work rose
by 9 tokens, or 16.7%, while the target-step proxy was identical.

## Decision

Rejected for runtime use. The reproducible analyzer and unit tests are retained
so future tapes can challenge the result, but the control plane continues to
preserve Pi's calibrated ordering and recorded completion order for ties.

The stronger research direction is genuine tree-aware verification with
per-branch confidence and one parallel target pass. That requires an engine
capability/protocol extension and real verifier timing; serial reordering is not
a substitute.

Reproduce one row with:

```sh
python examples/d3_tape_ablation.py \
  --tape /private/path/pattern-learning.json \
  --actor-model deepseek-v4-pro \
  --drafter-model deepseek-v4-flash \
  --tokenizer deepseek-ai/DeepSeek-V3 \
  --revision e815299b0bcbac849fa540c768ef21845365c9eb \
  --limits 28 \
  --orderings completion,prefix-consensus
```
