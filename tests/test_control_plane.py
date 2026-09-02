from __future__ import annotations

import asyncio
import math
import unittest
from collections.abc import AsyncIterator

import httpx
from fastapi import FastAPI

from self_speculation import (
    BoundaryDraftFeedback,
    BoundaryDraftStore,
    CandidateBundleBuilder,
    ControlRequestClosedError,
    DraftBundle,
    DraftReceipt,
    EngineCapabilities,
    EngineCacheReuseError,
    InferenceRequest,
    SelfSpeculationControlPlane,
    SnapshotForkRunner,
    StreamChunk,
    TokenLogprob,
    install_self_speculation_routes,
)


def encode(text: str) -> list[int]:
    return list(text.encode("utf-8"))


class CandidateBundleBuilderTest(unittest.IsolatedAsyncioTestCase):
    async def test_builds_ranked_concrete_calls_with_one_target_tokenizer(self) -> None:
        builder = CandidateBundleBuilder(
            tokenizer=encode,
            max_draft_tokens=12,
        )

        bundle = await builder.build(
            {
                "version": 2,
                "request_id": "actor-request",
                "format": "tagged_json",
                "boundary": "<tool_call>",
                "max_draft_tokens": 8,
                "model": {"id": "tiny"},
                "candidates": [
                    {
                        "id": "read-a",
                        "action_identity": {
                            "version": 1,
                            "predicted_action_id": "read-a",
                            "execution_action_id": "covering-read",
                            "projected": True,
                        },
                        "sources": ["drafter", "pattern-aware"],
                        "provenance": [{"proposalID": "p", "actionID": "a"}],
                        "tool_call": {
                            "name": "read",
                            "arguments": {"path": "a.txt"},
                        },
                        "score": {"conditional_probability": 0.9},
                    },
                    {
                        "id": "read-b",
                        "sources": ["drafter"],
                        "tool_call": {
                            "name": "read",
                            "arguments": {"path": "b.txt"},
                        },
                    },
                ],
            }
        )

        self.assertEqual(bundle.request_id, "actor-request")
        self.assertEqual(len(bundle.drafts), 2)
        self.assertEqual(
            [draft.metadata["candidate_id"] for draft in bundle.drafts],
            ["read-a", "read-b"],
        )
        self.assertEqual(
            bundle.drafts[0].metadata["sources"],
            ("drafter", "pattern-aware"),
        )
        self.assertEqual(
            bundle.drafts[0].metadata["action_identity"],
            {
                "version": 1,
                "predicted_action_id": "read-a",
                "execution_action_id": "covering-read",
                "projected": True,
            },
        )
        self.assertEqual(bundle.drafts[0].tool_calls[0].name, "read")
        self.assertLessEqual(len(bundle.drafts[0].token_ids), 8)
        self.assertEqual(
            bundle.drafts[0].boundary.token_ids,
            tuple(encode("<tool_call>")),
        )

    async def test_rejects_unbounded_or_malformed_candidate_payloads(self) -> None:
        builder = CandidateBundleBuilder(tokenizer=encode, max_candidates=1)
        with self.assertRaisesRegex(ValueError, "server limit"):
            await builder.build(
                {
                    "request_id": "x",
                    "candidates": [
                        {"tool_call": {"name": "a"}},
                        {"tool_call": {"name": "b"}},
                    ],
                }
            )
        with self.assertRaisesRegex(ValueError, "tool_call"):
            await builder.build(
                {"request_id": "x", "candidates": [{"id": "bad"}]}
            )
        with self.assertRaisesRegex(ValueError, "predicted_action_id"):
            await builder.build(
                {
                    "request_id": "x",
                    "candidates": [
                        {
                            "id": "predicted",
                            "action_identity": {
                                "predicted_action_id": "different",
                                "execution_action_id": "different",
                                "projected": False,
                            },
                            "tool_call": {"name": "read"},
                        }
                    ],
                }
            )


class RecordingBundleFeedback:
    name = "recording-bundles"

    def __init__(self) -> None:
        self.bundles: list[DraftBundle] = []
        self.cleared: list[str] = []

    async def submit_bundle(self, bundle: DraftBundle) -> DraftReceipt:
        self.bundles.append(bundle)
        return DraftReceipt(
            request_id=bundle.request_id,
            registered=True,
            draft_token_count=max(len(draft.token_ids) for draft in bundle.drafts),
            details={"candidate_count": len(bundle.drafts)},
        )

    async def clear(self, request_id: str) -> None:
        self.cleared.append(request_id)


class GatedBundleFeedback(RecordingBundleFeedback):
    def __init__(self) -> None:
        super().__init__()
        self.slow_started = asyncio.Event()
        self.release_slow = asyncio.Event()

    async def submit_bundle(self, bundle: DraftBundle) -> DraftReceipt:
        if bundle.request_id == "slow":
            self.slow_started.set()
            await self.release_slow.wait()
        return await super().submit_bundle(bundle)


class ForkEngine:
    name = "fork-engine"
    capabilities = EngineCapabilities(
        prompt=True,
        logprobs=True,
        prefix_cache=True,
        cache_read_reporting=True,
        max_context_tokens=32,
    )

    def __init__(self) -> None:
        self.requests: list[InferenceRequest] = []

    async def prompt_token_count(self, request: InferenceRequest) -> int:
        return len(request.prompt or "")

    async def stream(
        self, request: InferenceRequest
    ) -> AsyncIterator[StreamChunk]:
        self.requests.append(request)
        yield StreamChunk(
            text='{"name":"write","arguments":{"path":"out.txt"}}</tool_call>',
            token_ids=(1, 2),
            usage={
                "prompt_tokens": len(request.prompt or ""),
                "cache_read_tokens": 16,
            },
            logprobs=(
                TokenLogprob(token="write", logprob=-0.2),
                TokenLogprob(token="call", logprob=-0.4),
            ),
        )


class AgreeingForkEngine(ForkEngine):
    async def stream(
        self, request: InferenceRequest
    ) -> AsyncIterator[StreamChunk]:
        self.requests.append(request)
        yield StreamChunk(
            text='{"name":"read","arguments":{"path":"in.txt"}}</tool_call>',
            usage={"cache_read_tokens": 16},
        )


class ParallelForkEngine(ForkEngine):
    async def stream(
        self, request: InferenceRequest
    ) -> AsyncIterator[StreamChunk]:
        self.requests.append(request)
        yield StreamChunk(
            text='{"name":"read","arguments":{"path":"a.txt"}}</tool_call>',
            usage={"cache_read_tokens": 16},
        )
        yield StreamChunk(
            text=(
                '<tool_call>{"id":"call-b","name":"read",'
                '"arguments":{"path":"b.txt"}}</tool_call>'
            ),
            finish_reason="tool_calls",
        )


class GatedForkEngine(ForkEngine):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def stream(
        self, request: InferenceRequest
    ) -> AsyncIterator[StreamChunk]:
        self.requests.append(request)
        self.started.set()
        await self.release.wait()
        yield StreamChunk(
            text='{"name":"write","arguments":{"path":"out.txt"}}</tool_call>',
            usage={"cache_read_tokens": 16},
        )


class CacheMissForkEngine(ForkEngine):
    async def stream(
        self, request: InferenceRequest
    ) -> AsyncIterator[StreamChunk]:
        self.requests.append(request)
        yield StreamChunk(
            text='{"name":"read","arguments":{}}</tool_call>',
            usage={"prompt_tokens": len(request.prompt or "")},
        )


class SelfSpeculationControlPlaneTest(unittest.IsolatedAsyncioTestCase):
    def fixture(self):
        feedback = RecordingBundleFeedback()
        engine = ForkEngine()
        runner = SnapshotForkRunner(
            engine,
            encode,
            max_draft_tokens=32,
        )
        plane = SelfSpeculationControlPlane(
            feedback,
            CandidateBundleBuilder(tokenizer=encode, max_draft_tokens=32),
            fork_runner=runner,
        )
        return plane, feedback, engine

    async def test_merges_external_sources_and_self_fork_into_one_bundle(self) -> None:
        plane, feedback, engine = self.fixture()
        await plane.submit_candidates(candidate_payload())
        receipt = await plane.fork(fork_payload())

        self.assertEqual(receipt.details["candidate_count"], 2)
        self.assertEqual(len(feedback.bundles), 2)
        merged = feedback.bundles[-1]
        self.assertEqual(
            [draft.metadata["sources"] for draft in merged.drafts],
            [("drafter", "pattern-aware"), ("self-speculation",)],
        )
        self.assertTrue(
            str(merged.drafts[-1].metadata["candidate_id"]).startswith("self:")
        )
        self.assertEqual(
            receipt.details["bundle"]["candidates"][0]["action_identities"],
            (
                {
                    "version": 1,
                    "predicted_action_id": "external",
                    "execution_action_id": "external",
                    "projected": False,
                },
            ),
        )
        self.assertEqual(
            receipt.details["bundle"]["candidates"][0]["provenance"],
            ({"proposalID": "p", "actionID": "a"},),
        )
        observed = receipt.details["bundle"]["candidates"][-1]
        self.assertEqual(
            observed["tool_calls"],
            (
                {
                    "name": "write",
                    "arguments": {"path": "out.txt"},
                    "index": 0,
                    "format": "tagged_json",
                },
            ),
        )
        self.assertEqual(observed["fork"]["decoded_tokens"], 2)
        self.assertEqual(
            observed["fork"]["logprobs"],
            {
                "token_count": 2,
                "mean": -0.30000000000000004,
                "minimum": -0.4,
                "tool_name": {
                    "token_count": 1,
                    "matched_calls": 1,
                    "minimum": -0.2,
                    "minimum_probability": math.exp(-0.2),
                },
            },
        )
        self.assertEqual(
            observed["fork"]["context_budget"],
            {
                "prompt_tokens": 25,
                "max_context_tokens": 32,
                "max_output_tokens": 7,
            },
        )
        self.assertEqual(
            observed["fork"]["cache"],
            {
                "policy": "required",
                "configured": True,
                "reported": True,
                "verified": True,
                "reused_tokens": 16,
                "prompt_tokens": 25,
                "hit_rate": 16 / 25,
            },
        )
        self.assertGreaterEqual(observed["fork"]["total_ms"], 0)
        self.assertEqual(
            engine.requests[0].prompt,
            "PROMPTobserved<tool_call>",
        )
        self.assertEqual(
            engine.requests[0].extra,
            {"logprobs": True, "top_logprobs": 1},
        )
        self.assertEqual(engine.requests[0].max_tokens, 7)

        await plane.clear("actor-request")
        self.assertEqual(feedback.cleared, ["actor-request"])

    async def test_rejects_a_declared_cache_that_did_not_hit(self) -> None:
        runner = SnapshotForkRunner(CacheMissForkEngine(), encode)

        with self.assertRaisesRegex(EngineCacheReuseError, "no KV-cache reuse"):
            await runner.run(fork_payload())

    async def test_preserves_a_parallel_fork_as_one_complete_candidate(self) -> None:
        feedback = RecordingBundleFeedback()
        plane = SelfSpeculationControlPlane(
            feedback,
            CandidateBundleBuilder(tokenizer=encode, max_draft_tokens=64),
            fork_runner=SnapshotForkRunner(
                ParallelForkEngine(),
                encode,
                max_draft_tokens=64,
            ),
        )

        receipt = await plane.fork(fork_payload())

        self.assertEqual(receipt.details["candidate_count"], 1)
        observed = receipt.details["bundle"]["candidates"][0]
        self.assertEqual(
            observed["tool_calls"],
            (
                {
                    "name": "read",
                    "arguments": {"path": "a.txt"},
                    "index": 0,
                    "format": "tagged_json",
                },
                {
                    "name": "read",
                    "arguments": {"path": "b.txt"},
                    "index": 1,
                    "format": "tagged_json",
                    "call_id": "call-b",
                },
            ),
        )
        self.assertEqual(
            [(call.name, call.arguments) for call in feedback.bundles[-1].drafts[0].tool_calls],
            [("read", {"path": "a.txt"}), ("read", {"path": "b.txt"})],
        )

    async def test_closes_the_rendered_reasoning_envelope_before_the_probe(self) -> None:
        plane, _, engine = self.fixture()
        engine.capabilities = EngineCapabilities(
            prompt=True,
            logprobs=True,
            prefix_cache=True,
            cache_read_reporting=True,
            max_context_tokens=64,
        )
        payload = fork_payload()
        payload["context"]["provider_payload"]["prompt"] = "PROMPT<think>\n"
        payload["snapshot"] = {
            "generated_text": "reason",
            "reasoning": "reason",
            "chunk_count": 1,
            "output_chunk_count": 1,
        }

        await plane.fork(payload)

        self.assertEqual(
            engine.requests[0].prompt,
            "PROMPT<think>\nreason\n</think>\n\n<tool_call>",
        )

    async def test_deduplicates_self_agreement_by_complete_draft_content(self) -> None:
        feedback = RecordingBundleFeedback()
        engine = AgreeingForkEngine()
        plane = SelfSpeculationControlPlane(
            feedback,
            CandidateBundleBuilder(tokenizer=encode, max_draft_tokens=32),
            fork_runner=SnapshotForkRunner(
                engine,
                encode,
                max_draft_tokens=32,
            ),
        )

        await plane.submit_candidates(candidate_payload())
        receipt = await plane.fork(fork_payload())

        self.assertEqual(receipt.details["candidate_count"], 1)
        merged = feedback.bundles[-1].drafts[0]
        self.assertEqual(
            merged.metadata["sources"],
            ("drafter", "pattern-aware", "self-speculation"),
        )
        self.assertEqual(merged.metadata["source_count"], 3)
        self.assertEqual(len(merged.metadata["candidate_ids"]), 2)

    async def test_clear_fences_a_late_fork_without_blocking_cleanup(self) -> None:
        feedback = RecordingBundleFeedback()
        engine = GatedForkEngine()
        plane = SelfSpeculationControlPlane(
            feedback,
            CandidateBundleBuilder(tokenizer=encode),
            fork_runner=SnapshotForkRunner(engine, encode),
        )
        fork = asyncio.create_task(plane.fork(fork_payload()))
        await engine.started.wait()

        await plane.clear("actor-request")
        engine.release.set()

        with self.assertRaises(ControlRequestClosedError):
            await fork
        self.assertEqual(feedback.bundles, [])
        self.assertEqual(feedback.cleared, ["actor-request"])
        with self.assertRaises(ControlRequestClosedError):
            await plane.submit_candidates(candidate_payload())

    async def test_slow_feedback_does_not_serialize_unrelated_request_ids(self) -> None:
        feedback = GatedBundleFeedback()
        plane = SelfSpeculationControlPlane(
            feedback,
            CandidateBundleBuilder(tokenizer=encode),
        )
        slow_payload = {**candidate_payload(), "request_id": "slow"}
        fast_payload = {**candidate_payload(), "request_id": "fast"}
        slow = asyncio.create_task(plane.submit_candidates(slow_payload))
        await feedback.slow_started.wait()

        fast = await asyncio.wait_for(
            plane.submit_candidates(fast_payload), timeout=0.1
        )
        feedback.release_slow.set()
        await slow

        self.assertEqual(fast.request_id, "fast")
        self.assertEqual(
            [bundle.request_id for bundle in feedback.bundles],
            ["fast", "slow"],
        )

    async def test_installs_the_agent_facing_http_contract(self) -> None:
        plane, feedback, _ = self.fixture()
        app = FastAPI()
        self.assertTrue(install_self_speculation_routes(app, plane))
        self.assertFalse(install_self_speculation_routes(app, plane))
        client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://control",
        )

        candidates = await client.post(
            "/self-speculation/candidates", json=candidate_payload()
        )
        fork = await client.post("/self-speculation/fork", json=fork_payload())
        clear = await client.post(
            "/self-speculation/clear",
            json={"request_id": "actor-request"},
        )

        self.assertEqual(candidates.status_code, 200)
        self.assertEqual(fork.status_code, 200)
        self.assertEqual(fork.json()["details"]["candidate_count"], 2)
        self.assertEqual(clear.status_code, 200)
        self.assertEqual(feedback.cleared, ["actor-request"])
        await client.aclose()

    async def test_clear_returns_target_verification_telemetry(self) -> None:
        store = BoundaryDraftStore(max_draft_tokens=28)
        plane = SelfSpeculationControlPlane(
            BoundaryDraftFeedback(store),
            CandidateBundleBuilder(tokenizer=encode),
        )
        app = FastAPI()
        install_self_speculation_routes(app, plane)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://control",
        ) as client:
            payload = {**candidate_payload(), "prompt_token_count": 0}
            submitted = await client.post(
                "/self-speculation/candidates",
                json=payload,
            )
            self.assertEqual(submitted.status_code, 200)
            proposal = store.offer("actor-request", encode("<tool_call>"))
            self.assertIsNotNone(proposal)
            assert proposal is not None
            accepted = min(5, len(proposal.token_ids))
            store.observe_acceptance("actor-request", accepted)

            cleared = await client.post(
                "/self-speculation/clear",
                json={"request_id": "actor-request"},
            )

        self.assertEqual(cleared.status_code, 200)
        self.assertEqual(
            cleared.json()["verification"],
            expect_verification(len(proposal.token_ids), accepted, "external"),
        )


def candidate_payload():
    return {
        "version": 2,
        "request_id": "actor-request",
        "format": "tagged_json",
        "boundary": "<tool_call>",
        "candidates": [
            {
                "id": "external",
                "action_identity": {
                    "version": 1,
                    "predicted_action_id": "external",
                    "execution_action_id": "external",
                    "projected": False,
                },
                "sources": ["drafter", "pattern-aware"],
                "provenance": [{"proposalID": "p", "actionID": "a"}],
                "tool_call": {
                    "name": "read",
                    "arguments": {"path": "in.txt"},
                },
            }
        ],
    }


def fork_payload():
    return {
        "request_id": "actor-request",
        "model": {"id": "tiny"},
        "context": {
            "provider_payload": {"model": "tiny", "prompt": "PROMPT"}
        },
        "snapshot": {
            "generated_text": "observed",
            "content": "observed",
            "chunk_count": 1,
            "output_chunk_count": 1,
        },
        "options": {
            "forced_prefix": "<tool_call>",
            "decoder": "tagged_json",
            "draft_format": "tagged_json",
            "draft_boundary": "<tool_call>",
            "max_tokens": 64,
            "max_draft_tokens": 20,
            "temperature": 0,
            "require_logprobs": True,
        },
    }


def expect_verification(
    drafted: int,
    accepted: int,
    candidate_id: str,
) -> dict:
    return {
        "num_spec_steps": 1,
        "num_draft_tokens": drafted,
        "num_accepted_draft_tokens": accepted,
        "num_rejected_draft_tokens": drafted - accepted,
        "draft_acceptance_rate": accepted / drafted,
        "mean_acceptance_length": 1 + accepted,
        "per_step_drafted": [drafted],
        "per_step_accepted": [accepted],
        "steps": [{
            "candidate_index": 0,
            "candidate_id": candidate_id,
            "candidate_ids": [candidate_id],
            "sources": ["drafter", "pattern-aware"],
            "drafted_tokens": drafted,
            "accepted_tokens": accepted,
            "rejected_tokens": drafted - accepted,
        }],
        "unresolved_proposals": 0,
        "unresolved_draft_tokens": 0,
    }


if __name__ == "__main__":
    unittest.main()
