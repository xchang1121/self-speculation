# Causal D3 dynamic-cap ablation — 2026-08-26

This experiment asks whether newly connected target-verification feedback is
already sufficient to replace the fixed K=28 action-draft cap with a simple
online controller.

## Candidate policy

The tested policy is Hugging Face Transformers' documented persistent assisted
generation heuristic: if every offered draft token is accepted, increase the
next lookahead by two; otherwise decrease it by one. It is attractive here
because it needs only the accepted/proposed counts now returned by
`DraftVerificationOutcome`; it does not invent comparable confidence scores for
Drafter and PatternAware.

The Pi bridge can only select a cap for a later Actor request after the earlier
request clears, so the experiment applies the heuristic once per completed
request rather than inside one generation loop. Every tape starts cold. Within
each tape, opportunities are sorted by recorded Actor sequence and the policy
sees no future target action.

## Data and simulator

The replay uses the same three useful mock/SWE-style tapes, DeepSeek V3
tokenizer revision `e815299b0bcbac849fa540c768ef21845365c9eb`, `tagged_json`
formatter, and serial `BoundaryDraftStore` verifier model as the fixed-length
ablation. The 12 Actor action opportunities contain 416 target tokens and 21
distinct Drafter candidates. Results measure verifier work and target-step
proxies, not wall time.

`examples/d3_dynamic_cap_ablation.py` records every causal cap and verification
observation. `examples/d3_tape_ablation.py` now exposes the shared typed
opportunity builder so the fixed and dynamic runs cannot silently parse or
tokenize the tapes differently.

## Result

### Production starting point: K=28, bounds 4–28

| Policy | Mean K | Proposed | Accepted | Rejected | Acceptance | Target steps saved |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed K=28 | 28.00 | 198 | 144 | 54 | 72.7% | 138 |
| HF request-level heuristic | 27.50 | 198 | 144 | 54 | 72.7% | 138 |

The heuristic is completely inert at the verifier boundary. It lowers the
nominal cap after rejection, but the next concrete actions are already shorter
than that cap, so not one proposed, accepted, rejected, or saved-step count
changes. The per-tape cap traces were `28,28,27`, `28,28,27`, and
`28,28,27,28,27,26`.

### Sensitivity: K=20 cold start, ceiling 28

| Policy | Mean K | Proposed | Accepted | Rejected | Target steps saved |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed K=20 | 20.00 | 186 | 132 | 54 | 128 |
| HF request-level heuristic | 21.25 | 188 | 134 | 54 | 130 |
| Fixed K=28 production baseline | 28.00 | 198 | 144 | 54 | 138 |

Starting lower lets the heuristic recover two useful tokens without increasing
rejection, but it remains strictly dominated by the existing K=28 default,
which accepts ten more tokens and saves eight more target-step proxies with the
same rejected work.

## Decision

Reject the request-level HF heuristic for production. It adds persistent policy
state and asynchronous feedback ordering without improving any K=28 metric.
The causal analyzer and tests are retained so larger tapes can overturn this
decision without reimplementing the experiment.

This does not invalidate dynamic speculative decoding generally. Transformers'
confidence-based dynamic lookahead operates inside a token-generation loop,
while vLLM's current dynamic K addresses changing batch size. SVIP and
SpecDec++ likewise use draft-token uncertainty or a trained acceptance
predictor. The present external action candidates have neither token logprobs
on all sources nor target verification wall time, and a request-level `K−1`
update is too coarse to affect their short serialized bodies.

Primary references:

- [Transformers generation configuration](https://huggingface.co/docs/transformers/main_classes/text_generation)
- [Hugging Face dynamic speculation lookahead](https://huggingface.co/blog/dynamic_speculation_lookahead)
- [SVIP](https://arxiv.org/abs/2411.18462)
- [SpecDec++](https://arxiv.org/abs/2405.19715)
- [vLLM dynamic speculative decoding](https://docs.vllm.ai/en/latest/features/speculative_decoding/dynamic_speculative_decoding/)

Reproduce with:

```sh
python examples/d3_dynamic_cap_ablation.py \
  --tape /private/path/deepseek-mock-deterministic.json \
  --tape /private/path/deepseek-mock-success.json \
  --tape /private/path/pattern-learning.json \
  --actor-model deepseek-v4-pro \
  --drafter-model deepseek-v4-flash \
  --tokenizer deepseek-ai/DeepSeek-V3 \
  --revision e815299b0bcbac849fa540c768ef21845365c9eb \
  --initial-cap 28 --min-cap 4 --max-cap 28
```
