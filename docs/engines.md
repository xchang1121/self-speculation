# Engine compatibility and D3 setup

The streaming fork (D1) only needs an engine adapter. Injecting the decoded
tool action back into the main request (D3) additionally needs a stable request
ID, exact model token IDs, and a candidate path that is verified by the target
model. OpenAI API compatibility alone does not provide that path.

## Capability matrix

| Runtime | Streaming fork | Verified D3 injection | Integration |
| --- | --- | --- | --- |
| vLLM server or in-process | Yes | Yes | custom proposer plus HTTP or worker RPC |
| Transformers in-process | Yes | Yes | assisted-decoding candidate generator |
| SGLang server | Yes | Yes, on a compatible plugin build | NGRAM worker plugin plus corpus control routes |
| `llama-cpp-python` in-process | Yes | Yes | native `draft_model` callback |
| remote `llama-server` | Yes | No external action-injection API | OpenAI-compatible stream only |
| Hugging Face TGI | Yes | No external action-injection API | OpenAI-compatible stream only |
| custom runtime | Yes | When it implements target verification | `InferenceEngine` and `DraftFeedback` protocols |

The remote llama.cpp server and TGI may run their own built-in speculative
decoding. That is independent of this library's fork result: without an API for
request-scoped external candidates, the decoded action cannot enter their
verifier. Supporting that path would require a server fork or a new upstream
endpoint.

## Shared draft wiring

All verified integrations consume the same boundary-relative `DraftRequest`.
For a native engine, the engine itself can also be the feedback object:

```python
from self_speculation import (
    ForkController,
    ToolCallDraftBuilder,
    default_draft_boundary,
    format_tool_call_draft,
)

draft_builder = ToolCallDraftBuilder(
    formatter=format_tool_call_draft,
    tokenizer=engine.tokenize_continuation,
    boundary_resolver=default_draft_boundary,
    prompt_length_resolver=engine.prompt_token_count,
    max_draft_tokens=20,
)

controller = ForkController(
    engine,
    fork_builder,
    decoder_factory,
    fork_engine=fork_engine,
    draft_feedback=engine,
    draft_builder=draft_builder,
)
```

`fork_builder`, `decoder_factory`, and the tokenizer must use the same model
tool-call format. A mismatch is safe because the target rejects wrong draft
tokens, but it eliminates the speedup.

## Transformers

Install the optional runtime and construct an ordinary causal language model:

```bash
python -m pip install -e ".[transformers]"
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from self_speculation import BoundaryDraftStore, TransformersEngine

model_id = "YOUR_MODEL"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto").eval()
store = BoundaryDraftStore(max_draft_tokens=20)
engine = TransformersEngine(
    model,
    tokenizer,
    draft_store=store,
    max_draft_tokens=20,
)
```

The adapter streams `generate()` from a worker thread. When the boundary is
reached, `TransformersBoundaryCandidateGenerator` offers the remaining action
tokens through Transformers assisted decoding; the target model still scores
and accepts or rejects each candidate. D3 currently requires batch size one,
`num_beams=1`, and `use_cache=True`.

Use a separate model/engine for `fork_engine` when true concurrent main and
fork execution is required. The unit suite exercises a real tiny
`GPT2LMHeadModel` on Transformers 5.15.1 and confirms identical output with
fewer target forward calls for an accepted draft.

## llama-cpp-python

The in-process Python binding exposes the draft callback needed for external
candidates:

```bash
python -m pip install -e ".[llama-cpp]"
```

```python
from llama_cpp import Llama
from self_speculation import (
    BoundaryDraftStore,
    LlamaCppBoundaryDraftModel,
    LlamaCppPythonEngine,
)

store = BoundaryDraftStore(max_draft_tokens=20)
draft_model = LlamaCppBoundaryDraftModel(store, max_tokens=20)
llama = Llama(
    model_path="YOUR_MODEL.gguf",
    draft_model=draft_model,
    logits_all=True,
)
engine = LlamaCppPythonEngine(llama, draft_model=draft_model)
```

The exact `LlamaCppBoundaryDraftModel` object must be passed while constructing
`Llama`; installing it afterward does not enable the logits used for target
verification. One `Llama` object has mutable KV state and is serialized by the
adapter, so use a second `Llama` instance for a concurrent fork.

`LlamaCppEngine` is the separate remote `llama-server` adapter. The documented
server API provides built-in draft model and n-gram options, but not a
request-scoped external `draft_tokens` field, so that adapter remains D1-only.
See the official [llama.cpp speculative decoding guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md)
and [server reference](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md).

## SGLang

SGLang D3 uses the official general-plugin hook registry and the NGRAM target
verification path. Install this package in the same environment as every
SGLang process, then launch a plugin-compatible source build:

```bash
python -m pip install -e ".[http]"
export SGLANG_PLUGINS=self_speculation
python -m sglang.launch_server \
  --model-path YOUR_MODEL \
  --speculative-algorithm NGRAM \
  --speculative-num-draft-tokens 20
```

Connect the OpenAI endpoint and the root-level control routes separately:

```python
from transformers import AutoTokenizer
from self_speculation import (
    SGLangEngine,
    SGLangHTTPDraftFeedback,
    ToolCallDraftBuilder,
    default_draft_boundary,
    format_tool_call_draft,
)

engine = SGLangEngine("http://127.0.0.1:30000/v1", prefix_cache=True)
feedback = SGLangHTTPDraftFeedback("http://127.0.0.1:30000")
tokenizer = AutoTokenizer.from_pretrained("YOUR_MODEL")

draft_builder = ToolCallDraftBuilder(
    formatter=format_tool_call_draft,
    tokenizer=lambda text: tokenizer.encode(text, add_special_tokens=False),
    boundary_resolver=default_draft_boundary,
    max_draft_tokens=20,
)
```

`SGLangEngine` sends the library request ID as SGLang's `rid`. If the client
does not supply a prompt token count, the worker derives it from the active
request's exact `origin_input_ids`, avoiding BOS and chat-template drift. The
client tokenizer is still required for exact action and boundary token IDs.

The integration contract was checked against SGLang main commit
`7de80e566cc04c14a97d35ffd7270bb60186e9ba`. PyPI release `0.5.10.post1` does
not yet contain the general plugin hooks and external-corpus control API used
here; use a compatible source revision. See SGLang's official
[plugin guide](https://github.com/sgl-project/sglang/blob/main/docs/docs/hardware-platforms/plugin.mdx)
and [speculative decoding guide](https://github.com/sgl-project/sglang/blob/main/docs_new/docs/advanced_features/speculative_decoding.mdx).

The `/add_external_corpus` and `/remove_external_corpus` routes carry
request-scoped token IDs and can alter inference execution. Keep them on a
trusted network or protect them with an authenticated reverse proxy.

## TGI

`TGIEngine` supports D1 through TGI's OpenAI-compatible streaming endpoint.
TGI documents Medusa and n-gram speculation configured at server startup, but
does not expose request-scoped external draft candidates. Its repository is
also archived and in maintenance mode, so this project does not maintain a TGI
server fork. See the official [TGI speculation documentation](https://huggingface.co/docs/text-generation-inference/conceptual/speculation)
and [TGI repository](https://github.com/huggingface/text-generation-inference).

## Verification scope

The repository has unit and contract tests for every path. Transformers is
also exercised against a real local model implementation. vLLM, SGLang, and
llama.cpp GPU/server end-to-end tests require their optional runtimes, model
weights, and suitable hardware and are therefore deployment checks rather than
part of the default test suite.
