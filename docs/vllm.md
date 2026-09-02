# vLLM integration

This integration combines five pieces:

- vLLM's experimental `custom_class` speculative proposer;
- a narrow request-ID hook around `GPUModelRunner.propose_draft_token_ids`;
- a paired `vllm.general_plugins` entry point that installs worker RPC methods
  on every pipeline rank;
- four worker `collective_rpc` methods for single/bundle register, clear, and
  status;
- an opt-in `vllm.endpoint_plugins` entry point for tokenized drafts, concrete
  candidates, cleanup, and status.

The package does not depend on vLLM at import time. Incompatible vLLM V1
internals fail during proposer startup instead of silently routing a draft by
batch position.

## Install in the vLLM environment

Install this project in every environment/image used by the API frontend and
workers:

```bash
python -m pip install -e ".[http,server]"
```

For a normal wheel or package installation, replace the editable path with the
published artifact or repository URL.

## Start an OpenAI-compatible server

The endpoint and worker plugins deliberately share the entry-point name
`self_speculation`, so one explicit vLLM allowlist entry enables the HTTP and
worker halves together. Start the server with that allowlist and the custom
proposer:

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

On PowerShell, set the environment variable with:

```powershell
$env:VLLM_PLUGINS = "self_speculation"
```

`num_speculative_tokens` is the maximum D3 proposal length per verification
step. `SELF_SPECULATION_INJECT_WINDOW` optionally controls how many main-stream
body tokens may appear after the tool boundary before a registered draft is
considered stale; its default is `200`.

Check that the plugin and proposer are active:

```bash
curl http://127.0.0.1:8000/self-speculation/status
```

At least one worker result should have `"status":"ok"`. Pipeline stages that
do not own the proposer report `"status":"skipped"` and are expected.

The endpoint exposes `POST /self-speculation/drafts`, `/draft-bundles`,
`/candidates`, and `/clear`. The concrete-candidate route tokenizes every
boundary-relative tool-call body with the target tokenizer exposed by vLLM's
engine client, then registers one ordered bundle. Registration waits for the
matching main request to become active for a bounded interval, covering the
normal race between the OpenAI request and its control update.

`POST /self-speculation/clear` returns observed per-request verification when
the proposer saw at least one registered request:

```json
{
  "status": "cleared",
  "request_id": "actor-request-id",
  "verification": {
    "num_spec_steps": 2,
    "num_draft_tokens": 12,
    "num_accepted_draft_tokens": 9,
    "num_rejected_draft_tokens": 3,
    "draft_acceptance_rate": 0.75,
    "mean_acceptance_length": 5.5,
    "steps": [
      {
        "candidate_index": 0,
        "candidate_id": "drafter-0",
        "drafted_tokens": 8,
        "accepted_tokens": 7,
        "rejected_tokens": 1
      },
      {
        "candidate_index": 1,
        "candidate_id": "pattern-0",
        "drafted_tokens": 4,
        "accepted_tokens": 2,
        "rejected_tokens": 2
      }
    ],
    "unresolved_proposals": 0,
    "unresolved_draft_tokens": 0
  }
}
```

The worker that owns the custom proposer supplies this object; pipeline workers
without a proposer remain `skipped`. Sequence reconciliation observes a step
when vLLM next invokes the proposer. If the request ends immediately after an
offer, cleanup reports that last offer as unresolved instead of treating it as
a rejection. This telemetry changes no candidate ordering or proposal length.

## Agent sidecar for unified candidates and self-fork

The vLLM endpoint plugin can accept external candidates directly, but it does
not observe a separate agent's Actor stream and therefore does not expose
`/fork`. Run the portable control plane on a private sidecar port when the
agent should submit Drafter/PatternAware candidates and automatically start a
D1 fork from its first Actor delta:

```python
from fastapi import FastAPI
from transformers import AutoTokenizer

from self_speculation import (
    CandidateBundleBuilder,
    SelfSpeculationControlPlane,
    SnapshotForkRunner,
    VLLMEngine,
    VLLMHTTPDraftFeedback,
    install_self_speculation_routes,
)

MODEL = "YOUR_MODEL"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
fork_engine = VLLMEngine("http://127.0.0.1:8000/v1", prefix_cache=True)
feedback = VLLMHTTPDraftFeedback("http://127.0.0.1:8000")


def encode(text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def render(request) -> str:
    return tokenizer.apply_chat_template(
        list(request.messages),
        tools=list(request.tools) or None,
        tokenize=False,
        add_generation_prompt=True,
    )


control_plane = SelfSpeculationControlPlane(
    feedback,
    CandidateBundleBuilder(encode, max_candidates=64, max_draft_tokens=28),
    fork_runner=SnapshotForkRunner(
        fork_engine,
        encode,
        prompt_renderer=render,
        max_draft_tokens=28,
    ),
)
app = FastAPI()
install_self_speculation_routes(app, control_plane)
```

Serve this app on (for example) `127.0.0.1:8010`, point the agent bridge at
that URL, and select its `sidecar` fork transport. The sidecar sends the merged
bundle to vLLM through `/draft-bundles`; vLLM remains the only target verifier.
Use an application lifespan hook to close `fork_engine` and `feedback` during
shutdown.

If only external concrete actions are needed, point the agent directly at the
vLLM server and disable its sidecar fork. A `provider` transport is a separate
contract: the serving provider must explicitly interpret the injected
`self_speculation` request object and fork only the authoritative Actor stream.
Stock OpenAI-compatible request parsing and this endpoint plugin do not turn
unknown request fields into a fork automatically. Drafter requests can still
contribute external candidates, but are never self-forked.

## Connect a controller over HTTP

The tokenizer must be exactly compatible with the served model. The same
encoder is used for the predicted action body and the model-format boundary.

```python
import asyncio
import uuid

from transformers import AutoTokenizer

from self_speculation import (
    ForkController,
    InferenceRequest,
    PrefixForkBuilder,
    ToolCallDraftBuilder,
    VLLMEngine,
    VLLMHTTPDraftFeedback,
    default_decoder,
    default_draft_boundary,
    format_tool_call_draft,
)


MODEL = "YOUR_MODEL"
tokenizer = AutoTokenizer.from_pretrained(MODEL)


def encode(text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


async def main() -> None:
    engine = VLLMEngine(
        "http://127.0.0.1:8000/v1",
        prefix_cache=True,
    )
    feedback = VLLMHTTPDraftFeedback("http://127.0.0.1:8000")
    controller = ForkController(
        engine,
        PrefixForkBuilder(
            forced_prefix="<tool_call>",
            max_tokens=128,
            temperature=0.0,
        ),
        # The server continues after the forced prefix and does not echo it.
        lambda: default_decoder(
            "tagged_json",
            initial_text="<tool_call>",
        ),
        draft_feedback=feedback,
        draft_builder=ToolCallDraftBuilder(
            formatter=format_tool_call_draft,
            tokenizer=encode,
            boundary_resolver=default_draft_boundary,
            max_draft_tokens=28,
        ),
    )

    try:
        result = await controller.run(
            InferenceRequest(
                prompt="A fully rendered model prompt goes here",
                model=MODEL,
                request_id=uuid.uuid4().hex,
                max_tokens=512,
                temperature=0.0,
            )
        )
        print(result.tool_calls)
        print(result.draft_receipt)
    finally:
        await feedback.aclose()
        await engine.aclose()


asyncio.run(main())
```

`VLLMEngine` places the external `InferenceRequest.request_id` in both main and
fork request bodies. The worker bridge uniquely maps vLLM's wrapped internal ID
back to that external ID and keeps an alias until cleanup. Use a fresh,
high-entropy request ID for every controller run.

For chat inputs, `PrefixForkBuilder` also needs a `prompt_renderer` that applies
the identical chat template used by vLLM. Alternatively, render chat to a raw
prompt before constructing `InferenceRequest`.

## In-process AsyncLLM

When the application owns a vLLM `AsyncLLM`/`AsyncLLMEngine` object, use:

```python
engine = VLLMNativeEngine(async_llm, sampling_params_factory=...)
feedback = VLLMCollectiveRPCDraftFeedback(async_llm)
```

Configure that vLLM instance with the same `custom_class` speculative config.
The rest of the controller and draft-builder setup is identical. Both sync and
async `collective_rpc` methods are supported; synchronous calls run in a worker
thread so they do not block the controller event loop.

## Model-format choices

The forced prefix, decoder branch, and draft formatter boundary must describe
the same model format. Built-in combinations include:

| Model output family | Decoder name | Typical forced boundary |
| --- | --- | --- |
| Tagged/Hermes JSON | `tagged_json` | `<tool_call>` |
| Qwen XML function form | `qwen_xml` | `<tool_call>` |
| DeepSeek DSML | `deepseek_dsml` | `<｜DSML｜tool_calls>` |
| DeepSeek V3/R1 special tokens | `deepseek_v3` | `<｜tool▁calls▁begin｜>` |
| Pythonic calls | `pythonic` | `<|python_tag|>` |

When a forced prefix includes more than the top-level boundary, set the
decoder's `initial_text` to the complete forced text. For an unsupported model,
register a parser and provide matching formatter/boundary callbacks.

## Original SPORK endpoint compatibility

`SporkHTTPDraftFeedback` speaks the original `/spork/set_tokens` and
`/spork/clear` wire protocol. That endpoint clears global proposer state and
historically registers at batch index zero, so it is suitable for reproducing a
single-request SPORK setup but not for concurrent production traffic. Prefer
the request-scoped plugin above for new deployments.

## Security

The `/self-speculation/*` routes alter inference behavior. vLLM's API-key
middleware may only protect standard API prefixes such as `/v1`; do not assume
it protects plugin routes. Keep these endpoints private or enforce
authentication and an allowlist at a reverse proxy. Only install and allowlist
plugins that you trust.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| `404` under `/self-speculation` | package installed in frontend environment and `VLLM_PLUGINS=self_speculation` |
| all status rows are `skipped` | server uses the `VLLMBoundaryProposer` custom speculative config |
| `active vLLM request not found` | use `VLLMEngine`, or ensure another client sends a matching `request_id` body field; candidate registration retries only for its configured bounded interval |
| ambiguous external request ID | generate unique IDs; do not reuse prefix-related IDs concurrently |
| boundary-token validation error | use the served model tokenizer with `add_special_tokens=False` |
| draft submitted but no injection | main request did not reach the boundary, diverged from the draft, or exceeded the injection window |
| import or hook error at startup | installed vLLM does not expose the targeted V1 custom-proposer interface |

The implementation follows the current official vLLM
[custom proposer](https://docs.vllm.ai/en/stable/features/speculative_decoding/)
and [endpoint plugin](https://docs.vllm.ai/en/stable/design/endpoint_plugins/)
interfaces. Both are evolving extension surfaces, so pin and integration-test
the vLLM version used in production.
