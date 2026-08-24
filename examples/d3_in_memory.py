"""Runnable D1 + D3 demonstration without an external inference server."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

from self_speculation import (
    BoundaryDraftStore,
    DraftReceipt,
    DraftRequest,
    EngineCapabilities,
    ForkController,
    InferenceRequest,
    PrefixForkBuilder,
    StreamChunk,
    ToolCallDraftBuilder,
    default_decoder,
    default_draft_boundary,
    format_tool_call_draft,
)


def encode(text: str) -> list[int]:
    return list(text.encode("utf-8"))


class SignalingFeedback:
    """Demo feedback bridge; real engines use an HTTP/RPC adapter."""

    name = "demo-feedback"

    def __init__(self, store: BoundaryDraftStore) -> None:
        self.store = store
        self.ready = asyncio.Event()

    async def submit(self, draft: DraftRequest) -> DraftReceipt:
        receipt = self.store.register(draft)
        self.ready.set()
        return receipt

    async def clear(self, request_id: str) -> None:
        self.store.clear(request_id)


class DemoEngine:
    """Two deterministic streams standing in for one shared inference engine."""

    name = "demo-engine"
    capabilities = EngineCapabilities(
        prompt=True,
        token_ids=True,
        prefix_cache=True,
        draft_feedback=True,
    )

    def __init__(
        self,
        store: BoundaryDraftStore,
        feedback: SignalingFeedback,
    ) -> None:
        self.store = store
        self.feedback = feedback
        self.proposed_tokens: tuple[int, ...] = ()

    async def stream(self, request: InferenceRequest) -> AsyncIterator[StreamChunk]:
        if request.request_id.endswith(":fork"):
            # PrefixForkBuilder already forced "<tool_call>" into this request.
            yield StreamChunk(
                text=(
                    '\n{"name":"weather","arguments":{"city":"Shanghai"}}'
                    "</tool_call>"
                )
            )
            return

        # D1 starts the fork immediately after this first useful main delta.
        yield StreamChunk(reasoning="Need weather data. ", token_ids=(42,))
        await self.feedback.ready.wait()

        # The main request now reaches the tool-call boundary. The engine-side
        # proposer receives the full sequence and returns only the draft body.
        boundary_tokens = tuple(encode("<tool_call>"))
        sequence = (100, 42, *boundary_tokens)  # one prompt token, then output
        proposal = self.store.offer(
            request.request_id,
            sequence,
            sequence_length=len(sequence),
        )
        if proposal is None:
            raise RuntimeError("the demo draft was not offered")
        self.proposed_tokens = proposal.token_ids
        proposed_text = bytes(proposal.token_ids).decode("utf-8")
        yield StreamChunk(
            text="<tool_call>" + proposed_text,
            token_ids=boundary_tokens + proposal.token_ids,
            finish_reason="tool_calls",
        )


async def main() -> None:
    store = BoundaryDraftStore(max_draft_tokens=256)
    feedback = SignalingFeedback(store)
    engine = DemoEngine(store, feedback)
    controller = ForkController(
        engine,
        PrefixForkBuilder(forced_prefix="<tool_call>"),
        lambda: default_decoder(
            "tagged_json",
            initial_text="<tool_call>",
        ),
        draft_feedback=feedback,
        draft_builder=ToolCallDraftBuilder(
            formatter=format_tool_call_draft,
            tokenizer=encode,
            boundary_resolver=default_draft_boundary,
            prompt_length_resolver=lambda request: 1,
            max_draft_tokens=256,
        ),
    )

    result = await controller.run(
        InferenceRequest(prompt="What is the weather?", request_id="demo-turn")
    )
    call = result.tool_calls[0]
    print(
        json.dumps(
            {
                "tool_call": {"name": call.name, "arguments": call.arguments},
                "draft_registered": bool(
                    result.draft_receipt and result.draft_receipt.registered
                ),
                "draft_tokens_offered": len(engine.proposed_tokens),
                "store_after_run": {
                    "active_requests": store.snapshot().active_requests,
                    "injections": store.snapshot().injections,
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
