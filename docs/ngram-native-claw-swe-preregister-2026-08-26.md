# Native n-gram speculative decoding Claw-SWE preregistration (2026-08-26)

Status: frozen before sending any request from the I29 manifest to either
inference server.

## Candidate and motivation

I29 tests llama.cpp's native `ngram-simple` prompt-lookup proposer. It searches
already available target-token history and verifies the retrieved continuation
with the same target model. It does not launch another autoregressive model,
does not execute a predicted tool, and does not lower any action-confidence
threshold.

This is distinct from I25-I28. Those rounds generated an additional same-model
D1/D2 continuation. I29 drafts only from tokens already present in the prompt
or target output. It is motivated by Prompt Lookup Decoding, LLMA, REST, and
the native n-gram implementations now shipped by Transformers, vLLM, and
llama.cpp.

A retrospective discovery replay selected the family, not the holdout result.
Across the three existing DeepSeek mock tapes, using the already known unified
Drafter plus one PatternAware exact hit as the baseline, causal historical
action fallback changed `K=28` serial target steps from 254 to 230. It added 26
proposed tokens, all 26 accepted, with no additional rejected token. Those
observed tapes are excluded from the formal I29 decision.

## Frozen holdout

The private, model-output-free manifest is
`claw-swe-i29-ngram-cases-20260826.json` (128,513 bytes, SHA-256
`97e340f3b53ffed0faa611286f0dbf28988e9cfa3f7188dcd1ac1ef73529c7f7`).
It uses `TokenRhythm/Claw-SWE-Bench`, config `lite`, split `test`, revision
`ca9da7416154a31015f43df71dcf742c6725b312`, and the unchanged I27 system
prompt and read/bash/edit/write schemas.

The twelve request hashes have zero overlap with the thirty-one unique I25,
I26, and I27 requests:

| Case | Language | Request SHA-256 |
|---|---|---|
| `apache__lucene-13170` | Java | `3a69ebbecd62f8061f45ffa0287ec9660288ea093c50a058adb6051e9ceecaa9` |
| `projectlombok__lombok-3042` | Java | `b2eee31319e86307b310f213a6ec204f2a285fe9ec59df72b003808fb3f4d44c` |
| `gin-gonic__gin-1957` | Go | `cb063b7b40d21cdd3339a4774190013ecaa80e1c29db459b2c5d4f6cfc065e87` |
| `hashicorp__terraform-35611` | Go | `47c92f487fe9b6bb7cb15b7c4b7a286fcc1ca45b0f90fff2471338b8e3ff62c3` |
| `nushell__nushell-12950` | Rust | `0989261824a1bd9d86550b639e1c37678e583b41adcaa62a9e419c3dab8611e9` |
| `facebook__docusaurus-10130` | JS/TS | `6bdacc1cc866a2ac6b4745c38e6f0f7eaef191063c8474651cf6b67419a25002` |
| `mrdoob__three.js-25687` | JS/TS | `1162f3faf031ddc88445e806c98a9fbaab25942a6dcb0aa7bc0062a7b9a0c31a` |
| `jqlang__jq-2598` | C/C++ | `0a4b8d7f85e4f14c74edfa28595ef057c8e07126a74e18d0919e5c9254c5cd5b` |
| `fluent__fluentd-4030` | Ruby | `f2b5e783052a0e978d99df47a943fc879e1fb02f054507576930ef1df9a6ab5a` |
| `laravel__framework-51195` | PHP | `f7e59bd3ab15e9368697645fc15c33776c1c2382ddcfa82cf3e516700e5c1147` |
| `phpoffice__phpspreadsheet-3940` | PHP | `544308fe8f8b14277fead7908d81988b4bbb546cad927e90c6c91d090b3b28bb` |
| `sphinx-doc__sphinx-8551` | Python | `3d6d55da9a0bf18dbf371d35d6de2368c3097818f6df7aba3e8f4d7c21053778` |

The manifest contains no model answer, action, reasoning, patch, or gold test.
Only public problem text, fixed request scaffolding, integrity hashes, and
source metadata are present.

## Frozen A/B

Both arms use Qwen3-1.7B Q8 GGUF revision
`90862c4b9d2787eaed51d12237eafdfe7c5f6077`, tokenizer revision
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, official llama.cpp build 10615 /
`f280b2698`, Vulkan, one 8192-token slot, flash attention, 64-token prompt-cache
reuse, greedy temperature 0, seed 42, a 3,072-token maximum, and the same
`</tool_call>` inclusive stop.

- control: `--spec-type none`;
- treatment: `--spec-type ngram-simple
  --spec-ngram-simple-size-n 8 --spec-ngram-simple-size-m 28
  --spec-ngram-simple-min-hits 1`.

The n-gram key of eight avoids the very short structured-output matches called
out by current vLLM issue reports; the 28-token continuation equals the
evidence-backed D3 cap. The values are frozen and will not be tuned on this
holdout.

The two fresh servers remain loaded simultaneously. Each receives the same
fixed excluded warm-up. Cases run in manifest order; even rows execute control
then treatment and odd rows treatment then control. Each server therefore sees
the same case order while pair order alternates to reduce thermal/order bias.
Only the main decode runs: no D1/D2 probe is charged or allowed to perturb the
comparison.

The recorder snapshots llama.cpp Prometheus counters immediately after warm-up
and after the final pair. The target-forward proxy is
`n_decode_total + spec_decode_num_drafts_total`: ordinary target decode calls
plus speculative verification calls. Draft/accepted counters come directly
from the treatment server.

## Frozen gates

All product gates must pass:

1. **Validity:** 12/12 pairs complete without an error; at least eight pairs
   produce a non-empty main output.
2. **Losslessness:** control and treatment token IDs, decoded text, stop type,
   and truncation flag are identical for every pair.
3. **Useful speculation:** treatment drafts at least 28 tokens and accepts at
   least 50% of all drafted tokens.
4. **Target work:** pooled treatment target-forward proxy is at most 95% of
   control.
5. **Decode time:** pooled server-reported `predicted_ms` and paired wall-time
   sum are each at most 95% of control; the median paired `predicted_ms` ratio
   is at most 0.95.
6. **Tail safety:** no treatment case exceeds 1.25x its control
   `predicted_ms` (zero-time pairs are invalid).

The report will also disclose prompt/cache tokens, generated tokens, draft
acceptance, per-case ratios, parseable tool-call counts, and any cap-limited
outputs. Tool-call correctness is not a gate because both arms must be
token-identical; this experiment measures lossless decoding work, not whether
Qwen solves the SWE issue.

If any gate fails, native n-gram remains an archived analyzer/result only and
no Pi, protocol, control-plane, or engine-default patch is accepted. If all
gates pass, product integration still requires a separate minimal design review
because vLLM, llama.cpp, and the custom external-candidate proposer expose
different multi-proposer contracts.
