# D2 think-mode re-probe preregistration (2026-08-26)

## Question

Does SPORK's published D2 mechanism recover exact tool calls that a single D1
fork misses, without spending enough extra probe work to erase the usable
decode runway?

This is deliberately separate from the earlier remote-API D2 upper bounds.
Those DeepSeek action turns exposed little or no safe intermediate reasoning.
The new test uses a local thinking model, where D2's required growing CoT
prefix and token log-probabilities actually exist.

## Frozen implementation reference

- SPORK repository: `baihuajun24/spork` at
  `31d5ab6f0740d5b5aa26e6a745dc97bcff5139a3`.
- Active `d1_d2` behavior: first probe after token 1; at most five probes;
  retry after 50 more main tokens, snapped to a sentence boundary (or forced
  after 30 additional tokens); greedily decode at most 100 probe tokens.
- Gate: minimum selected-token probability over positions 2 through 21 must be
  at least `0.90`; the first parseable probe that passes commits.
- Verification: exact normalized tool name and arguments. Name-only matches do
  not count.

## Fixed local runtime

- Model: `Qwen/Qwen3-1.7B-GGUF`, `Q8_0`, revision
  `90862c4b9d2787eaed51d12237eafdfe7c5f6077`.
- Prompt tokenizer: `Qwen/Qwen3-1.7B`, revision
  `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`.
- Inference: official `llama.cpp` Windows package build `10615`, commit
  `f280b2698`, installed through the hash-verified winget manifest.
- Generation: thinking enabled, seed 42, greedy decode, `</tool_call>` stop.
  Greedy decode matches the released SPORK runner even though the Qwen model
  card recommends sampling for general reasoning quality.

## Inputs and leakage controls

Use every Actor request from the three established recordings:

- `deepseek-mock-deterministic.json`
- `deepseek-mock-success.json`
- `pattern-learning.json`

Requests are canonical-hash deduplicated. Their recorded messages and tool
schemas are rendered for Qwen; their DeepSeek action is only a portability
reference. D1 and D2 are scored against the *same local main decode*, matching
SPORK's accept/reject definition. Raw prompts and model outputs remain in the
private recording directory; only hashes and aggregates enter git.

## Policies

1. `D1`: the token-1 probe dispatches whenever it parses; strict verification
   happens when the local main action is known.
2. `D1+D2`: apply the frozen retry schedule and confidence gate above; commit
   the first qualifying parseable probe.
3. `D2 oracle`: report whether any recorded retry exactly matches the local
   main action. This is diagnostic only and can never justify product code.

The recorder captures main token timestamps, probe wall time, selected-token
log-probabilities, generated token count, and prompt-cache accounting. Runway
is an optimistic no-contention counterfactual:

`main end - (snapshot token time + probe wall time)`.

It must not be reported as measured concurrent latency.

## Acceptance gates (fixed before generation)

Product work is allowed only if all gates pass:

1. **Validity:** at least six deduplicated turns produce a parseable local main
   tool call.
2. **Recall:** D1+D2 has no fewer exact hits than D1, loses no D1 exact hit, and
   recovers at least one D1 miss.
3. **Precision:** exact-hit precision among dispatched D1+D2 probes is no lower
   than D1.
4. **Probe efficiency:** mean D1+D2 attempts per eligible turn is at most 2.0,
   and total generated probe tokens are at most 1.75 times D1's.
5. **Usable recovery:** every claimed recovered hit finishes with at least
   25 ms of optimistic main-decode runway.
6. **No threshold tuning:** the product decision uses only the preregistered
   `0.90` threshold. `0.85` and `0.95` may be reported as sensitivity checks,
   never substituted after seeing results.

If any gate fails, no D2 product patch is retained. The recorder/analyzer and
negative report may still be committed because they prevent repeating an
unsupported experiment.

## Scope boundary

This test can establish mechanism feasibility for a local think-mode inference
endpoint. It cannot overturn the existing negative result for ordinary remote
DeepSeek direct-action streams, and it cannot claim production latency gains
without a concurrent serving experiment.
