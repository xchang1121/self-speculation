# Native n-gram speculative decoding preflight ablation (2026-08-26)

## Decision

**Reject llama.cpp `ngram-simple` for this integration and do not consume the
frozen twelve-case I29 holdout.** A clean preflight on one already used I27
request changes the greedy token stream and tool action, accepts only 14.5% of
draft tokens, increases the target-forward proxy by 51.9%, and increases total
decode time by 57.6%.

No Pi, protocol, control-plane, inference adapter, or engine-default change is
made. The generic paired recorder remains because it provides a strict way to
test a future engine build or a genuinely different retrieval proposer without
reusing this result as a tuned threshold.

## Why this candidate was tested

Prompt Lookup Decoding and LLMA retrieve continuations from text already in
the target context, avoiding another autoregressive drafter. REST generalizes
the same idea to a retrieval datastore. Current Transformers, vLLM, and
llama.cpp releases expose native prompt/n-gram lookup paths:

- Prompt Lookup Decoding: https://github.com/apoorvumang/prompt-lookup-decoding
- LLMA: https://arxiv.org/abs/2304.04487
- REST: https://arxiv.org/abs/2311.08252
- Transformers assisted decoding:
  https://huggingface.co/docs/transformers/main/assisted_decoding
- vLLM n-gram proposer:
  https://github.com/vllm-project/vllm/blob/main/vllm/v1/spec_decode/ngram_proposer.py
- llama.cpp speculative decoding:
  https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md

This fit the local constraint better than I25-I28: reuse existing prompt/action
tokens and pay only verification, rather than generating another same-model
probe. The retrospective discovery screen was positive: after including the
known PatternAware exact hit, causal history fallback on the three DeepSeek
mock tapes reduced simulated `K=28` target steps from 254 to 230, while all 26
additional proposed tokens were accepted.

## Frozen configuration and unused holdout

The configuration and formal gates were committed before any I29 case request
as `89b8cc5`. The paired recorder and tests were then committed as `16b1c49`.

Both arms used Qwen3-1.7B Q8 revision
`90862c4b9d2787eaed51d12237eafdfe7c5f6077`, tokenizer revision
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, official llama.cpp build 10615 /
`f280b2698`, Vulkan, one 8192-token slot, flash attention, greedy seed 42, and
a 3,072-token cap. The treatment alone added:

```text
--spec-type ngram-simple
--spec-ngram-simple-size-n 8
--spec-ngram-simple-size-m 28
--spec-ngram-simple-min-hits 1
```

The frozen private manifest
`claw-swe-i29-ngram-cases-20260826.json` is 128,513 bytes with SHA-256
`97e340f3b53ffed0faa611286f0dbf28988e9cfa3f7188dcd1ac1ef73529c7f7`.
Its twelve request hashes have zero overlap with I25-I27. **None of those twelve
requests was sent to either server.** The holdout remains unconsumed because
the preflight failed multiple hard gates.

## Clean preflight

The recorder first proved that its excluded repetitive warm-up was token-exact,
that control generated zero drafts, and that treatment generated native
drafts. It then ran the already observed `google__gson-1100` I27 request on two
freshly started servers. Both arms cold-evaluated the same 1,829-token prompt;
prompt evaluation was 256.059 ms for control and 249.560 ms for treatment.
This removes the cache-state confound found in an earlier discarded harness
smoke.

Private recording:
`qwen3-ngram-native-smoke-fresh-20260826.json`, 208,829 bytes, SHA-256
`3ed3a620fb902998481e04b82aebabeb4856976d428fb2cdc878468d21434432`.

| Metric | Control | `ngram-simple` | Ratio / delta |
|---|---:|---:|---:|
| Generated tokens | 1,417 | 2,409 | +992 |
| Native draft tokens | 0 | 2,346 | -- |
| Accepted native draft tokens | 0 | 341 | 14.54% acceptance |
| Native verification steps | 0 | 84 | +84 |
| Target-forward proxy | 1,417 | 2,152 | **1.519x** |
| Server `predicted_ms` | 14,187.144 | 22,352.936 | **1.576x** |
| Paired wall time | 14,502.787 | 22,656.365 | **1.562x** |
| Parseable tool call | `edit` | `read` | changed |

The first token mismatch occurs at generated token 310. Control ultimately
calls `edit` on `src/main/java/com/example/Person.java`; treatment calls `read`
on `Demo.java`. Neither output is cap-truncated. Thus this is not merely a
different stopping boundary or text reconstruction issue.

Speculative decoding is intended to preserve the target distribution, but
bit-exact greedy output can still be lost in a concrete backend when batched
verification changes floating-point logits around a near tie. This report does
not infer the internal cause from one trace. The observable contract is enough:
an agent action changed, so the path cannot be enabled under the preregistered
losslessness gate.

## Gate result

| Frozen gate | Result |
|---|---|
| 12 complete / at least 8 non-empty | not run; stopped at preflight |
| Every pair token/text/stop exact | **fail** at token 310 |
| At least 28 drafts and at least 50% accepted | **fail** (`2346`, `14.54%`) |
| Target-forward ratio at most 0.95 | **fail** (`1.519`) |
| Pooled/median predicted and wall ratios at most 0.95 | **fail** (`1.576`, `1.562`) |
| No case above 1.25x predicted time | **fail** (`1.576x`) |

It would be invalid to tune key length, draft length, occurrence threshold, or
sampling flags against this observed action and then call the same manifest a
holdout. A materially different prompt-lookup implementation may be registered
as a new candidate only with a new preflight and unchanged action-safety gate.

## Verification and product consequence

The recorder independently captures both arms, alternates pair order, verifies
server metadata and warm-up behavior, snapshots native Prometheus counters,
writes each completed pair atomically, and recomputes all gates from the raw
recording. Its tests cover counter parsing, forward accounting, timing gates,
and output mismatch rejection.

Verification passed: 160 unit tests, `compileall`, `pip check`, smoke reanalysis,
and `git diff --check`.

No negative product patch exists to revert. I29 retains only the preregistration,
paired benchmark infrastructure, and this negative report. The next candidate
should stay inside the already lossless external `BoundaryDraftStore` verifier
path, where exact output equivalence has real-model tests, rather than enabling
this native proposer alongside it.
