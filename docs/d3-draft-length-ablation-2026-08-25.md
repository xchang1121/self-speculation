# D3 action-draft length ablation — 2026-08-25

This experiment selects a safer default D3 cap and evaluates whether the
current integration has enough feedback for an online dynamic-length policy.

## Method

`examples/d3_tape_ablation.py` decodes the strict Actor/Drafter tool calls from
the three useful private tapes, serializes each call with the current
`tagged_json` formatter, and tokenizes it with `deepseek-ai/DeepSeek-V3` revision
`e815299b0bcbac849fa540c768ef21845365c9eb`. The tokenizer is a reproducible
proxy for the mocked DeepSeek V4 endpoint; it is not asserted to be V4's exact
production tokenizer.

For each Actor action, candidates are deduplicated and ordered by recorded
completion time. The simulator mirrors `BoundaryDraftStore`: one candidate is
offered at a time, the target accepts the exact common prefix, samples one bonus
token after that prefix, and may then try a compatible fallback candidate. It
reports proposed/accepted tokens and target verification steps. It does not
convert those steps into wall time.

## Pooled result

The useful tapes contain 12 Actor action opportunities, 21 unique bundled
candidates, and 416 target action-body tokens under this tokenizer/formatter.

| K cap | Proposed | Accepted | Rejected | Acceptance | Target steps saved | Step reduction proxy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 48 | 48 | 0 | 100.0% | 48 | 11.5% |
| 8 | 96 | 76 | 20 | 79.2% | 76 | 18.3% |
| 12 | 144 | 102 | 42 | 70.8% | 102 | 24.5% |
| 20 | 186 | 132 | 54 | 71.0% | 128 | 30.8% |
| 24 | 194 | 140 | 54 | 72.2% | 136 | 32.7% |
| **28** | **198** | **144** | **54** | **72.7%** | **138** | **33.2%** |
| 32 | 198 | 144 | 54 | 72.7% | 138 | 33.2% |

Moving from K=20 to K=28 offers 12 additional tokens; all 12 match the Actor,
so rejected work stays at 54 tokens while the target-step proxy saves 10 more
steps. K=32, 40, 48, and 64 produce no additional proposal or saving on these
recordings. K=28 is therefore the smallest observed Pareto saturation point and
replaces 20 as the Pi/control-plane/store/Transformers-engine default. User
configuration and the engine's own hard K cap still take precedence.

## Dynamic policy decision

Do not add an online heuristic yet. `DraftReceipt.accepted_token_count` is a
registration-time field in the current side channel; registration occurs before
the target reaches the action boundary, so it is not acceptance feedback. The
tapes also contain neither D3 verification timing nor token-level draft
confidence. Treating missing receipt values as zero would drive a false feedback
loop and shrink correct drafts.

Recent approaches reinforce that a real policy needs real signals:

- [SVIP](https://arxiv.org/abs/2411.18462) uses draft-token distribution
  uncertainty to decide when to stop.
- [Learning to Draft](https://arxiv.org/abs/2603.01639) optimizes true
  draft-and-verify cycle time rather than acceptance alone.
- [vLLM dynamic speculative decoding](https://docs.vllm.ai/en/latest/features/speculative_decoding/dynamic_speculative_decoding/)
  adjusts K for concurrency, and current vLLM can expose
  [per-request acceptance metrics](https://docs.vllm.ai/en/latest/features/speculative_decoding/acceptance_metrics/)
  in the final usage chunk.

The next legitimate dynamic implementation should preserve those vLLM metrics
through the provider bridge (or expose an equivalent engine callback), then tune
K against measured verifier wall time and batch size. Until that feedback exists,
the fixed K=28 cap is the evidence-backed improvement and dynamic K remains
deferred rather than simulated in production.

Reproduce one tape with:

```sh
python examples/d3_tape_ablation.py \
  --tape /private/path/pattern-learning.json \
  --actor-model deepseek-v4-pro \
  --drafter-model deepseek-v4-flash \
  --tokenizer deepseek-ai/DeepSeek-V3 \
  --revision e815299b0bcbac849fa540c768ef21845365c9eb \
  --limits 4,8,12,20,24,28,32
```
