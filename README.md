# self-speculation

`self-speculation` extracts the reusable inference mechanisms from
[SPORK](https://github.com/baihuajun24/spork) into a small, engine-agnostic
Python library:

1. fork one speculative continuation from a running streaming inference;
2. decode that continuation into one or more structured tool calls;
3. optionally return the decoded action to the main engine as draft tokens for
   target-model verification (SPORK D3).

The authoritative main stream always continues independently. Forking, parsing,
and draft feedback are replaceable components, so the same controller can be
used with different serving engines and model-specific tool-call formats.
Self-speculation always forks the authoritative Actor request. External
Drafter or learned candidates may share its target-verification bundle, but
their inference requests are never self-forked.

See [the architecture guide](docs/architecture.md) for the complete D1/D3 data
flow and failure semantics, and the [engine compatibility guide](docs/engines.md)
for setup instructions and verified-injection support.

## What is included

- a dependency-free streaming engine protocol and normalized request/chunk
  models;
- concurrent main/fork orchestration with SPORK's first-output D1 trigger;
- prefix and callback-based fork request builders;
- structured-delta decoding plus nine built-in text parser branches;
- adapters for arbitrary callbacks, OpenAI-compatible servers, native vLLM,
  Transformers, and `llama-cpp-python` generation;
- portable callback/HTTP draft feedback, an agent-facing control plane, ordered
  multi-source draft bundles, and a request-scoped boundary store;
- verified D3 integrations for vLLM, Transformers, SGLang NGRAM, and
  `llama-cpp-python`;
- typed lifecycle events, best-effort acceleration failures, and strict modes.

Agent loops, tool execution, benchmark harnesses, datasets, and scheduling
policy are intentionally out of scope.

## Install and run the demonstration

Python 3.10 or newer is required.

```bash
python -m pip install -e ".[test]"
python examples/d3_in_memory.py
```

The example runs without an external model server. It demonstrates the full
flow: first-output fork, streaming tool-call decode, request-scoped draft
registration, boundary matching, one proposal, and cleanup.

Optional dependencies are deliberately separated:

| Extra | Install command | Purpose |
| --- | --- | --- |
| Core | `python -m pip install -e .` | custom/local engines and parsers |
| HTTP | `python -m pip install -e ".[http]"` | OpenAI-compatible and HTTP feedback clients |
| Server | `python -m pip install -e ".[server]"` | vLLM endpoint plugin routes |
| Transformers | `python -m pip install -e ".[transformers]"` | native Transformers streaming and verified D3 |
| llama.cpp Python | `python -m pip install -e ".[llama-cpp]"` | in-process llama.cpp streaming and verified D3 |
| Development | `python -m pip install -e ".[test]"` | complete test suite |

## Minimal streaming fork

This example uses a raw completion endpoint. The prompt must already use the
served model's chat/tool template.

```python
import asyncio

from self_speculation import (
    ForkController,
    InferenceRequest,
    PrefixForkBuilder,
    VLLMEngine,
    default_decoder,
)


async def main() -> None:
    async with VLLMEngine(
        "http://127.0.0.1:8000/v1",
        prefix_cache=True,
    ) as engine:
        controller = ForkController(
            engine,
            PrefixForkBuilder(
                forced_prefix="<tool_call>",
                max_tokens=128,
                temperature=0.0,
            ),
            # The forced prefix is normally not repeated by the server.
            lambda: default_decoder(
                "tagged_json",
                initial_text="<tool_call>",
            ),
        )
        result = await controller.run(
            InferenceRequest(
                prompt="A fully rendered model prompt",
                model="YOUR_MODEL",
                max_tokens=512,
            )
        )

    print(result.main.content)  # authoritative model output
    print(result.tool_calls)    # speculative fork prediction


asyncio.run(main())
```

Use `ForkController.stream()` instead of `run()` when an application needs
main chunks, fork chunks, completed calls, draft receipts, or failures as typed
events in real time. A separate `fork_engine` can be supplied, although using
the same prefix-caching engine is what enables SPORK's D1 cache reuse.

## Engine adapters

| Engine family | Adapter | Notes |
| --- | --- | --- |
| Any sync or async iterable | `CallableEngine` | maps strings, dictionaries, or native chunks |
| OpenAI-compatible SSE | `OpenAICompatibleEngine` | raw and chat completions; structured deltas |
| vLLM server | `VLLMEngine` | also forwards stable external request IDs for D3 |
| SGLang | `SGLangEngine` | forwards stable `rid`; optional NGRAM D3 plugin |
| Hugging Face TGI | `TGIEngine` | streaming fork only; no external D3 injection API |
| llama.cpp server | `LlamaCppEngine` | streaming fork only; no external D3 injection API |
| In-process vLLM | `VLLMNativeEngine` | adapts `AsyncLLM`/`AsyncLLMEngine` generation |
| In-process Transformers | `TransformersEngine` | native streaming; optional assisted-decoding D3 |
| In-process llama.cpp | `LlamaCppPythonEngine` | native streaming; optional draft-model D3 |
| Another runtime | `InferenceEngine` protocol | implement one async `stream(request)` method |

An adapter normalizes each provider update into `StreamChunk`, keeping visible
text, reasoning, token IDs, logprobs, structured tool deltas, and native data
separate. Engine-specific request options remain available through
`InferenceRequest.extra`.

## Tool-call parser branches

`default_decoder("auto")` tries the registered text branches in order and
locks onto the first one that produces a call. Provider-native structured tool
deltas are decoded directly and take precedence over text parsing.

| Decoder name | Supported shape or markers |
| --- | --- |
| `deepseek_dsml` | DeepSeek DSML tool-call markup |
| `deepseek_v3` | DeepSeek V3/R1 special-token function syntax |
| `qwen_xml` | Qwen XML function and parameter elements |
| `tagged_json` | Hermes/Qwen/SPORK JSON inside `<tool_call>` |
| `mistral_json` | Mistral `[TOOL_CALLS]` JSON |
| `llama_json` | Llama JSON after `<|python_tag|>` |
| `pythonic` | Python-like function calls, parsed without executing code |
| `xlam_json` | xLAM fenced, tagged, reasoning-prefixed, or bare JSON |
| `bare_json` | a bare JSON object or list |

Parsers are incremental: delimiters and JSON/XML fragments may cross arbitrary
transport chunks. Registering another branch does not require changing the
controller:

```python
registry = default_parser_registry()
registry.register("my_format", MyStreamingParser)
controller = ForkController(
    engine,
    fork_builder,
    lambda: registry.decoder("my_format"),
)
```

## D3 draft feedback

Decoding a fork predicts an action batch, but it only accelerates main-model token
generation when the engine verifies that prediction through its speculative
decoding path. `ToolCallDraftBuilder` formats and tokenizes calls after their
model-specific boundary; `DraftFeedback` atomically replaces the main request's
ordered candidate set. A single prediction uses the same path as a set of one.

| Feedback path | Intended use |
| --- | --- |
| `CallableDraftFeedback` | application callback or custom runtime |
| `BoundaryDraftFeedback` + `BoundaryDraftStore` | in-process engines and testing |
| `HTTPDraftFeedback` | portable request-scoped sidecar |
| `SporkHTTPDraftFeedback` | compatibility with original SPORK HTTP routes |
| `VLLMHTTPDraftFeedback` | remote vLLM endpoint plugin |
| `VLLMCollectiveRPCDraftFeedback` | in-process vLLM worker RPC |
| `SGLangHTTPDraftFeedback` | remote SGLang NGRAM plugin control plane |
| `TransformersEngine` | in-process assisted-decoding candidate injection |
| `LlamaCppPythonEngine` | in-process llama.cpp draft-model injection |

The store excludes prompt history, matches single- or multi-token boundaries,
checks any already-generated action prefix, and offers only the remaining
suffix. A ranked bundle falls through to the next distinct action after target
rejection; each candidate fires at most once. Incremental bundle replacement
preserves already-fired identities. The target engine still verifies every
proposed token, so disagreement falls back to ordinary target-model decoding.
Each verifier step is retained with its candidate ID and proposed, accepted,
and rejected token counts. Clearing a request returns this observed outcome;
the portable HTTP adapters expose it as an optional `verification` object.
An offer that cannot be reconciled before cleanup is reported separately as
unresolved rather than being guessed as accepted or rejected.

`DraftReceipt.accepted_token_count` belongs to the registration handshake and
is not necessarily target-model acceptance. Use the clear-time verification
outcome for policy tuning. Transformers reports its native `num_matches`
callback directly; boundary-only integrations reconcile an offer from the next
target sequence and keep a final unseen offer unresolved.

The candidate builder, boundary store, snapshot fork, and Transformers adapter
default to an action-draft cap of 28 tokens; explicit application and engine
caps still take precedence. Production preserves the submitted candidate order
and uses a fixed request-scoped cap. Real-model regression tests cover exact
output preservation through the target verifier.

### Agent-facing unified control plane

`CandidateBundleBuilder` converts ranked concrete tool calls from any agent
predictor with the target tokenizer. `SelfSpeculationControlPlane` combines
that external bundle with an optional D1 `SnapshotForkRunner`, deduplicates
identical complete drafts while merging provenance, and submits one ordered
`DraftBundle` to the target verifier.

The snapshot runner consumes the fork through its terminal stream boundary
before constructing a draft, so structured or textual parallel calls arriving
across multiple chunks remain one candidate rather than silently truncating to
the first call. Bundle observations preserve every call index/ID/format plus
candidate IDs, sources, proposal provenance, scores, timing, and logprobs for
the action runtime that consumes the receipt.

Before a textual fork is built, `ContinuationPlanner` matches the open CoT
envelope in the rendered Actor prompt and uses that envelope's own closer. It
also restores a hidden reasoning-to-content transition when an API exposes the
two fields separately. Signed or otherwise structured reasoning without a
matching text envelope is rejected; that path requires an engine-native fork
which preserves the provider state instead of fabricating text.

Prompt-side envelope detection is anchored to the active assistant-generation
tail, so an unmatched tag quoted earlier in user or history text cannot select
a closer. A closer already emitted in raw Actor output is never duplicated.

The selected tool format also supplies a name-aligned probe prefix. JSON/XML
schema text is forced through the character immediately before the tool name,
so D2 confidence is measured from generated tool-name tokens rather than from
framing such as a JSON `name` key. Probe receipts retain the attempt number and
the exact Actor snapshot size used by an external retry scheduler.

`SnapshotForkRunner` also defaults to `cache_policy="required"`. A declaration
that prefix caching is enabled is not enough: the adapter must expose the
fork's per-request cache-read token count and that count must be positive.
Native vLLM reads `RequestOutput.num_cached_tokens`; vLLM and SGLang HTTP
adapters request and normalize their usage reports. `prefer` and `off` remain
available for experiments, but their receipts do not claim a verified hit.

Its portable FastAPI routes are:

| Route | Purpose |
| --- | --- |
| `POST /self-speculation/candidates` | replace the current external candidate set |
| `POST /self-speculation/fork` | run one self-fork from a captured Actor snapshot |
| `POST /self-speculation/clear` | fence late updates, return observed verification, and clear verifier state |

All three operations use the same high-entropy Actor request ID. Candidate and
fork work is serialized only within that ID; unrelated requests remain
concurrent. Clear installs a bounded tombstone before waiting for in-flight
work, so a late fork cannot recreate a closed request. See the
[vLLM guide](docs/vllm.md#agent-sidecar-for-unified-candidates-and-self-fork)
for complete wiring.

Candidate-control version 2 may include `action_identity`. Its
`predicted_action_id` is also the candidate ID used by target verification;
`execution_action_id` separately identifies a lossless covering execution.
The control plane validates this distinction and preserves all identities when
equal token drafts merge, without using execution identity to collapse distinct
Actor-visible predictions.

Clear-time verification steps carry all merged `candidate_ids` and `sources`,
not only the primary draft ID. Clients can therefore calibrate token acceptance
for every contributing predictor without inferring attribution from bundle
order or confusing registration with target acceptance.

### vLLM quick start

Install the package in the vLLM frontend and worker environments, then opt in
to the paired endpoint/worker plugin and custom proposer. Both plugin entry
points use the name `self_speculation`, so one allowlist value enables both:

```bash
export VLLM_PLUGINS=self_speculation
vllm serve YOUR_MODEL \
  --enable-prefix-caching \
  --speculative-config '{
    "method": "custom_class",
    "model": "self_speculation.integrations.vllm.VLLMBoundaryProposer",
    "num_speculative_tokens": 28
  }'
```

The controller pairs `VLLMEngine` with `VLLMHTTPDraftFeedback`. See the
[vLLM integration guide](docs/vllm.md) for complete client code, native-engine
usage, model-format alignment, request-ID routing, security, and
troubleshooting.

The `/self-speculation/*` routes affect inference execution. Keep them on a
trusted network or protect them at a reverse proxy; do not assume that a vLLM
API key covers plugin routes outside `/v1`.

## Development

```bash
python -m unittest discover -s tests -v
python -m compileall -q src examples tests
python examples/d3_in_memory.py
```

The test suite covers controller concurrency and cleanup, every parser family,
all engine adapters, draft formatting/feedback/storage, vLLM proposal routing,
Transformers candidate verification, llama.cpp draft callbacks, SGLang NGRAM
hooks, worker RPC, and HTTP endpoints.

## Relationship to SPORK

This repository is a focused reorganization of
**SPORK: Self-Speculative Forking to Accelerate Agentic LLM Inference**. The
D1/D3 design, upstream MIT license, and pre-refactor Git history are retained
and attributed.

- Original SPORK repository: <https://github.com/baihuajun24/spork>
- Starting fork: <https://github.com/xchang1121/spork>
- Paper: <https://arxiv.org/abs/2607.03333>
- Additional attribution: [NOTICE.md](NOTICE.md)

If this work contributes to research results, cite the SPORK paper:

```bibtex
@misc{bai2026spork,
  title         = {SPORK: Self-Speculative Forking to Accelerate Agentic LLM Inference},
  author        = {Bai, Huajun and Lv, Weiwei and Zheng, Huichuan and Lu, Youyou and Shu, Jiwu},
  year          = {2026},
  eprint        = {2607.03333},
  archivePrefix = {arXiv},
  primaryClass  = {cs.DC}
}
```

## License

MIT. See [LICENSE](LICENSE).
