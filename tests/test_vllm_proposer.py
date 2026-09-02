from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from fastapi import FastAPI

from self_speculation import (
    DraftBoundary,
    DraftBundle,
    DraftRequest,
    SelfSpeculationEndpointPlugin,
    VLLMBoundaryProposer,
    VLLMCollectiveRPCDraftFeedback,
    VLLMIntegrationError,
    VLLMHTTPDraftFeedback,
    install_vllm_http_routes,
    install_vllm_request_id_hook,
    install_vllm_worker_rpc,
)


def bundle(*drafts: DraftRequest) -> DraftBundle:
    return DraftBundle(drafts[0].request_id, drafts)


class FakeRunner:
    def __init__(self, drafter, request_ids: list[str]) -> None:
        self.drafter = drafter
        self.input_batch = SimpleNamespace(req_ids=request_ids)

    def propose_draft_token_ids(self, value: str) -> str:
        return value


class VLLMRequestIdHookTest(unittest.TestCase):
    def test_routes_stable_batch_ids_and_installs_once(self) -> None:
        class Drafter:
            request_ids: tuple[str, ...] = ()

            def set_request_ids(self, request_ids: tuple[str, ...]) -> None:
                self.request_ids = request_ids

        drafter = Drafter()
        runner = FakeRunner(drafter, ["main", "main:fork"])
        self.assertTrue(install_vllm_request_id_hook(FakeRunner))
        self.assertFalse(install_vllm_request_id_hook(FakeRunner))
        self.assertEqual(runner.propose_draft_token_ids("result"), "result")
        self.assertEqual(drafter.request_ids, ("main", "main:fork"))

    def test_rejects_an_incompatible_runner(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "unsupported vLLM"):
            install_vllm_request_id_hook(type("Unsupported", (), {}))


class VLLMBoundaryProposerTest(unittest.TestCase):
    def proposer(self) -> VLLMBoundaryProposer:
        config = SimpleNamespace(
            speculative_config=SimpleNamespace(num_speculative_tokens=4)
        )
        with patch(
            "self_speculation.integrations.vllm.install_vllm_request_id_hook"
        ), patch("self_speculation.integrations.vllm.install_vllm_worker_rpc"):
            return VLLMBoundaryProposer(config)

    def test_proposes_for_matching_stable_id_after_batch_reordering(self) -> None:
        proposer = self.proposer()
        proposer.register_draft_bundle(
            "main", [[20, 21, 22]], [[10, 11]], 2
        )
        proposer.set_request_ids(("main:fork", "main"))

        proposals = proposer.propose(
            sampled_token_ids=[[7], [8]],
            num_tokens_no_spec=[3, 6],
            token_ids_cpu=[
                [90, 91, 7, 0, 0, 0],
                [90, 91, 5, 10, 11, 20],
            ],
        )

        self.assertEqual(proposals, [[], [21, 22]])
        self.assertEqual(proposer.status()["injections"], 1)

    def test_returns_empty_without_an_exact_request_mapping(self) -> None:
        proposer = self.proposer()
        proposer.register_draft_bundle("main", [[1]], [[9]])
        self.assertEqual(
            proposer.propose([[1]], [1], [[9]]),
            [[]],
        )

    def test_exposes_request_scoped_cleanup(self) -> None:
        proposer = self.proposer()
        result = proposer.register_draft_bundle("main", [[1, 2]], [[9]])
        self.assertTrue(result["registered"])
        self.assertTrue(proposer.clear_request("main")["removed"])
        self.assertEqual(proposer.clear_all()["removed"], 0)

    def test_uses_ranked_bundle_fallback_after_target_rejection(self) -> None:
        proposer = self.proposer()
        result = proposer.register_draft_bundle(
            "main",
            [[20, 21], [20, 99, 30]],
            [[10], [10]],
            2,
            candidate_metadata=[
                {"candidate_id": "first", "sources": ["actor-fork"]},
                {"candidate_id": "fallback", "sources": ["drafter"]},
            ],
        )
        proposer.set_request_ids(("main",))

        first = proposer.propose([[1]], [3], [[90, 91, 10]])
        fallback = proposer.propose(
            [[1]], [5], [[90, 91, 10, 20, 99]]
        )

        self.assertEqual(result["candidate_count"], 2)
        self.assertEqual(first, [[20, 21]])
        self.assertEqual(fallback, [[30]])
        self.assertEqual(proposer.status()["fallback_injections"], 1)
        cleared = proposer.clear_request("main")
        verification = cleared["verification"]
        self.assertEqual(verification["num_draft_tokens"], 2)
        self.assertEqual(verification["num_accepted_draft_tokens"], 1)
        self.assertEqual(verification["num_rejected_draft_tokens"], 1)
        self.assertEqual(verification["steps"][0]["candidate_id"], "first")
        self.assertEqual(verification["steps"][0]["sources"], ["actor-fork"])
        self.assertEqual(verification["unresolved_proposals"], 1)
        self.assertEqual(verification["unresolved_draft_tokens"], 1)
        self.assertEqual(proposer.status()["unresolved_draft_tokens"], 1)


class VLLMWorkerRPCBridgeTest(unittest.TestCase):
    def test_installs_registration_cleanup_and_status_methods(self) -> None:
        class Worker:
            pass

        proposer = VLLMBoundaryProposerTest().proposer()
        worker = Worker()
        worker.model_runner = SimpleNamespace(
            drafter=proposer,
            requests={
                "cmpl-main-0-random": SimpleNamespace(
                    prompt_token_ids=[9, 8, 7]
                )
            },
        )

        self.assertTrue(install_vllm_worker_rpc(Worker))
        self.assertFalse(install_vllm_worker_rpc(Worker))
        bundled = worker.self_speculation_register_draft_bundle(
            "main",
            [[1, 2], [1, 3]],
            [[9], [9]],
            None,
            [{"candidate_id": "a"}, {"candidate_id": "b"}],
        )
        status = worker.self_speculation_draft_status()
        proposer.set_request_ids(("cmpl-main-0-random",))
        before_new_boundary = proposer.propose([[1]], [3], [[9, 8, 7]])
        at_new_boundary = proposer.propose([[1]], [4], [[9, 8, 7, 9]])
        worker.model_runner.requests.clear()
        cleared = worker.self_speculation_clear_draft("main")

        self.assertTrue(bundled["registered"])
        self.assertEqual(bundled["candidate_count"], 2)
        self.assertEqual(bundled["internal_request_id"], "cmpl-main-0-random")
        self.assertEqual(status["active_requests"], 1)
        self.assertEqual(before_new_boundary, [[]])
        self.assertEqual(at_new_boundary, [[1, 2]])
        self.assertTrue(cleared["removed"])

    def test_skips_pipeline_workers_without_a_proposer(self) -> None:
        class Worker:
            pass

        install_vllm_worker_rpc(Worker)
        worker = Worker()
        worker.model_runner = SimpleNamespace()
        result = worker.self_speculation_register_draft_bundle(
            "x", [[1]], [[2]]
        )
        self.assertEqual(result["status"], "skipped")

    def test_rejects_ambiguous_external_request_ids(self) -> None:
        class Worker:
            pass

        proposer = VLLMBoundaryProposerTest().proposer()
        worker = Worker()
        worker.model_runner = SimpleNamespace(
            drafter=proposer,
            requests={
                "chatcmpl-shared-a": SimpleNamespace(prompt_token_ids=[1]),
                "chatcmpl-shared-b": SimpleNamespace(prompt_token_ids=[1]),
            },
        )
        install_vllm_worker_rpc(Worker)
        result = worker.self_speculation_register_draft_bundle(
            "shared", [[1]], [[2]], None
        )
        self.assertEqual(result["status"], "error")
        self.assertIn("ambiguous", result["error"])


class FakeAsyncEngineClient:
    def __init__(self, results=None) -> None:
        self.calls: list[tuple[str, float | None, tuple[object, ...]]] = []
        self.results = results

    async def collective_rpc(self, method, *, timeout=None, args=()):
        self.calls.append((method, timeout, args))
        if self.results is not None:
            return self.results
        if method.endswith("register_draft_bundle"):
            return [
                {"status": "ok", "registered": True, "draft_token_count": 2},
                {"status": "skipped", "reason": "no_boundary_proposer"},
            ]
        if method.endswith("draft_status"):
            return [{"status": "ok", "active_requests": 1}]
        return [{"status": "cleared"}, {"status": "skipped"}]


class FakeTokenizer:
    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return list(text.encode("utf-8"))


class ActivatingEngineClient(FakeAsyncEngineClient):
    def __init__(self) -> None:
        super().__init__()
        self.renderer = SimpleNamespace(tokenizer=FakeTokenizer())
        self.pending_registrations = 2

    async def collective_rpc(self, method, *, timeout=None, args=()):
        self.calls.append((method, timeout, args))
        if method.endswith("register_draft_bundle"):
            if self.pending_registrations:
                self.pending_registrations -= 1
                return [{"status": "pending", "reason": "request_not_active"}]
            return [
                {
                    "status": "ok",
                    "registered": True,
                    "draft_token_count": 4,
                    "candidate_count": len(args[1]),
                }
            ]
        return [{"status": "cleared"}]


class VerificationEngineClient(FakeAsyncEngineClient):
    async def collective_rpc(self, method, *, timeout=None, args=()):
        if method.endswith("clear_draft"):
            self.calls.append((method, timeout, args))
            return [
                {
                    "status": "cleared",
                    "verification": {
                        "num_spec_steps": 1,
                        "num_draft_tokens": 2,
                        "num_accepted_draft_tokens": 1,
                        "num_rejected_draft_tokens": 1,
                        "per_step_drafted": [2],
                        "per_step_accepted": [1],
                    },
                },
                {"status": "skipped", "reason": "no_boundary_proposer"},
            ]
        return await super().collective_rpc(
            method,
            timeout=timeout,
            args=args,
        )


class VLLMCollectiveRPCDraftFeedbackTest(unittest.IsolatedAsyncioTestCase):
    async def test_registers_and_clears_across_workers(self) -> None:
        client = VerificationEngineClient()
        feedback = VLLMCollectiveRPCDraftFeedback(client, timeout=7)
        draft = DraftRequest(
            request_id="main",
            token_ids=(20, 21),
            boundary=DraftBoundary(token_ids=(10, 11)),
            prompt_token_count=4,
        )
        receipt = await feedback.submit(bundle(draft))
        outcome = await feedback.clear("main")

        method, timeout, args = client.calls[0]
        self.assertEqual(method, "self_speculation_register_draft_bundle")
        self.assertEqual(timeout, 7)
        self.assertEqual(args, ("main", [[20, 21]], [[10, 11]], 4, [{}]))
        self.assertEqual(receipt.draft_token_count, 2)
        self.assertEqual(client.calls[1][0], "self_speculation_clear_draft")
        self.assertEqual(outcome.accepted_tokens if outcome else None, 1)
        self.assertEqual(outcome.rejected_tokens if outcome else None, 1)

    async def test_registers_an_ordered_bundle_across_workers(self) -> None:
        client = FakeAsyncEngineClient()
        feedback = VLLMCollectiveRPCDraftFeedback(client)
        bundle = DraftBundle(
            "main",
            (
                DraftRequest(
                    request_id="main",
                    token_ids=(20, 21),
                    boundary=DraftBoundary(token_ids=(10,)),
                    metadata={"candidate_id": "first", "sources": ["actor-fork"]},
                ),
                DraftRequest(
                    request_id="main",
                    token_ids=(20, 99),
                    boundary=DraftBoundary(token_ids=(10,)),
                    metadata={"candidate_id": "fallback"},
                ),
            ),
        )

        receipt = await feedback.submit(bundle)

        self.assertEqual(receipt.details["candidate_count"], 2)
        self.assertEqual(
            client.calls[0][2],
            (
                "main",
                [[20, 21], [20, 99]],
                [[10], [10]],
                None,
                [
                    {"candidate_id": "first", "sources": ["actor-fork"]},
                    {"candidate_id": "fallback"},
                ],
            ),
        )

    async def test_rejects_invalid_drafts_or_missing_proposers(self) -> None:
        feedback = VLLMCollectiveRPCDraftFeedback(
            FakeAsyncEngineClient(results=[{"status": "skipped"}])
        )
        with self.assertRaisesRegex(ValueError, "token_ids"):
            await feedback.submit(
                bundle(DraftRequest(request_id="x", text="raw"))
            )
        with self.assertRaisesRegex(ValueError, "boundary token_ids"):
            await feedback.submit(
                bundle(DraftRequest(request_id="x", token_ids=(1,)))
            )
        with self.assertRaisesRegex(VLLMIntegrationError, "no vLLM worker"):
            await feedback.submit(
                bundle(
                    DraftRequest(
                        request_id="x",
                        token_ids=(1,),
                        boundary=DraftBoundary(token_ids=(2,)),
                    )
                )
            )


class VLLMHTTPRoutesTest(unittest.IsolatedAsyncioTestCase):
    async def test_round_trips_feedback_through_fastapi_routes(self) -> None:
        app = FastAPI()
        engine_client = VerificationEngineClient()
        app.state.engine_client = engine_client
        self.assertTrue(install_vllm_http_routes(app))
        self.assertFalse(install_vllm_http_routes(app))

        client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://vllm",
        )
        feedback = VLLMHTTPDraftFeedback("http://vllm", client=client)
        receipt = await feedback.submit(
            bundle(
                DraftRequest(
                    request_id="main/one",
                    token_ids=(20, 21),
                    boundary=DraftBoundary(text="<tool_call>", token_ids=(10,)),
                    prompt_token_count=3,
                )
            )
        )
        status = await feedback.status()
        outcome = await feedback.clear("main/one")

        self.assertTrue(receipt.registered)
        self.assertEqual(status["worker_results"][0]["active_requests"], 1)
        self.assertEqual(
            engine_client.calls[-1][2],
            ("main/one",),
        )
        self.assertEqual(outcome.accepted_tokens if outcome else None, 1)
        await client.aclose()

    async def test_remote_feedback_submits_a_complete_bundle(self) -> None:
        app = FastAPI()
        engine_client = FakeAsyncEngineClient()
        app.state.engine_client = engine_client
        install_vllm_http_routes(app)
        client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://vllm",
        )
        feedback = VLLMHTTPDraftFeedback("http://vllm", client=client)
        bundle = DraftBundle(
            request_id="main",
            drafts=(
                DraftRequest(
                    request_id="main",
                    token_ids=(20, 21),
                    boundary=DraftBoundary(token_ids=(10,)),
                    metadata={"candidate_id": "first"},
                ),
                DraftRequest(
                    request_id="main",
                    token_ids=(20, 99),
                    boundary=DraftBoundary(token_ids=(10,)),
                    metadata={"candidate_id": "fallback"},
                ),
            ),
        )

        receipt = await feedback.submit(bundle)

        self.assertTrue(receipt.registered)
        self.assertEqual(receipt.details["candidate_count"], 2)
        self.assertTrue(engine_client.calls[0][0].endswith("register_draft_bundle"))
        self.assertEqual(
            [item["candidate_id"] for item in engine_client.calls[0][2][-1]],
            ["first", "fallback"],
        )
        await client.aclose()

    async def test_tokenizes_concrete_candidates_and_waits_for_actor_admission(self) -> None:
        app = FastAPI()
        engine_client = ActivatingEngineClient()
        app.state.engine_client = engine_client
        install_vllm_http_routes(
            app,
            registration_wait_timeout=0.2,
            registration_retry_interval=0.001,
        )
        client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://vllm",
        )

        response = await client.post(
            "/self-speculation/candidates",
            json={
                "request_id": "actor",
                "format": "tagged_json",
                "boundary": "<tool_call>",
                "max_draft_tokens": 20,
                "candidates": [
                    {
                        "id": "candidate-a",
                        "sources": ["drafter", "pattern-aware"],
                        "tool_call": {
                            "name": "read",
                            "arguments": {"path": "a.txt"},
                        },
                    }
                ],
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["details"]["candidate_count"], 1)
        self.assertEqual(engine_client.pending_registrations, 0)
        self.assertEqual(len(engine_client.calls), 3)
        _, _, args = engine_client.calls[-1]
        self.assertEqual(args[0], "actor")
        self.assertEqual(args[4][0]["candidate_id"], "candidate-a")
        self.assertEqual(args[4][0]["sources"], ("drafter", "pattern-aware"))
        self.assertTrue(args[1][0])
        self.assertTrue(args[2][0])
        await client.aclose()

    async def test_rejects_invalid_route_payloads(self) -> None:
        app = FastAPI()
        app.state.engine_client = FakeAsyncEngineClient()
        install_vllm_http_routes(app)
        client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://vllm",
        )
        response = await client.post(
            "/self-speculation/draft-bundles",
            json={"request_id": "x", "drafts": [{"token_ids": [1]}]},
        )
        self.assertEqual(response.status_code, 422)
        await client.aclose()

    async def test_official_endpoint_plugin_initializes_engine_state(self) -> None:
        app = FastAPI()
        engine_client = FakeAsyncEngineClient()
        plugin = SelfSpeculationEndpointPlugin()
        plugin.attach_router(app)
        await plugin.init_state(engine_client, app.state, SimpleNamespace())

        client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://vllm",
        )
        response = await client.get("/self-speculation/status")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["worker_results"][0]["active_requests"], 1)
        await client.aclose()


if __name__ == "__main__":
    unittest.main()
