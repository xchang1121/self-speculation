# Real-Transformers D3 ablation — 2026-08-25

This experiment verifies that the portable D3 path reduces real target-model
forward calls after the default K=28 alignment. It deliberately separates
engine integration from action-prediction quality.

## Method

`examples/transformers_d3_ablation.py` loads
[`hf-internal-testing/tiny-random-gpt2`](https://huggingface.co/hf-internal-testing/tiny-random-gpt2)
at revision `71034c5d8bde858ff824298bdedc65515b97d2b9` with PyTorch
`2.13.0+cpu` and Transformers `5.15.1`.

The baseline greedily generates 32 tokens. The ablation copies tokens 2–29
from that deterministic output into one request-scoped 28-token boundary draft.
This exact replay is an oracle for the D3 data path: it measures verifier
overhead and output preservation, not the probability that Drafter,
PatternAware, or a D1 fork predicts the action correctly. Both paths use the
same model, prompt, streamer, cache setting, and process. After one warm-up per
mode, seven alternating baseline/assisted pairs are measured with model forward
hooks and `perf_counter`.

## Result

| Metric | Baseline | D3 K=28 | Change |
| --- | ---: | ---: | ---: |
| Generated tokens | 32 | 32 | identical token IDs in 7/7 pairs |
| Median target forward calls | 32 | 4 | **−87.5%** |
| Median wall time | 163.043 ms | 24.229 ms | **6.729×** local speedup |

All eight registered drafts (one warm-up and seven measured runs) were injected;
the store proposed 224 tokens with zero divergent/stale drafts and returned to
zero active requests after cleanup.

The reproducible script now also consumes the native Transformers
`num_matches` callback and fails unless every exact-replay proposal is resolved
and fully accepted. Its JSON report includes the measured-request totals under
`verification`, separating target acceptance from registration receipts. A
seven-pair rerun on 2026-08-26 observed 196/196 accepted measured draft tokens,
zero rejected tokens, and zero unresolved proposals; the store total including
warm-up was 224/224. Output token IDs remained identical and target forward
calls remained 32 versus 4.

## Decision and boundary

Accepted as a real-engine integration gate. It confirms that the
`TransformersEngine` adapter reaches Transformers assisted decoding, target
verification preserves greedy output exactly, and K=28 collapses 28 serial
decode calls into batched verification in this controlled case.

The wall-time ratio is not a production estimate: the model is tiny, runs on
CPU, and receives an exact replay candidate. Production benefit must multiply
this verified D3 opportunity by actual candidate acceptance and include
Drafter/D1 overhead, concurrency, batching, and hardware effects. The tape
experiments provide the candidate-quality evidence; this experiment supplies
the missing real-verifier evidence.

Reproduce with:

```sh
python examples/transformers_d3_ablation.py \
  --model hf-internal-testing/tiny-random-gpt2 \
  --revision 71034c5d8bde858ff824298bdedc65515b97d2b9 \
  --repeats 7 \
  --max-new-tokens 32 \
  --max-draft-tokens 28
```
