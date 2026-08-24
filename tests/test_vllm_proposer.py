from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from fastapi import FastAPI

from self_speculation import (
    DraftBoundary,
    DraftRequest,
    VLLMBoundaryProposer,
    VLLMCollectiveRPCDraftFeedback,
    VLLMIntegrationError,
    VLLMHTTPDraftFeedback,
    install_vllm_http_routes,
    install_vllm_request_id_hook,
    install_vllm_worker_rpc,
)


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
        proposer.register_draft("main", [20, 21, 22], [10, 11], 2)
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
        proposer.register_draft("main", [1], [9])
        self.assertEqual(
            proposer.propose([[1]], [1], [[9]]),
            [[]],
        )

    def test_exposes_request_scoped_cleanup(self) -> None:
        proposer = self.proposer()
        result = proposer.register_draft("main", [1, 2], [9])
        self.assertTrue(result["registered"])
        self.assertTrue(proposer.clear_request("main")["removed"])
        self.assertEqual(proposer.clear_all()["removed"], 0)


class VLLMWorkerRPCBridgeTest(unittest.TestCase):
    def test_installs_registration_cleanup_and_status_methods(self) -> None:
        class Worker:
            pass

        proposer = VLLMBoundaryProposerTest().proposer()
        worker = Worker()
        worker.model_runner = SimpleNamespace(
            drafter=proposer,
            requests={
                "main": SimpleNamespace(prompt_token_ids=[9, 8, 7])
            },
        )

        self.assertTrue(install_vllm_worker_rpc(Worker))
        self.assertFalse(install_vllm_worker_rpc(Worker))
        registered = worker.self_speculation_register_draft(
            "main", [1, 2], [9], None
        )
        status = worker.self_speculation_draft_status()
        proposer.set_request_ids(("main",))
        before_new_boundary = proposer.propose([[1]], [3], [[9, 8, 7]])
        at_new_boundary = proposer.propose([[1]], [4], [[9, 8, 7, 9]])
        cleared = worker.self_speculation_clear_draft("main")

        self.assertTrue(registered["registered"])
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
        result = worker.self_speculation_register_draft("x", [1], [2])
        self.assertEqual(result["status"], "skipped")


class FakeAsyncEngineClient:
    def __init__(self, results=None) -> None:
        self.calls: list[tuple[str, float | None, tuple[object, ...]]] = []
        self.results = results

    async def collective_rpc(self, method, *, timeout=None, args=()):
        self.calls.append((method, timeout, args))
        if self.results is not None:
            return self.results
        if method.endswith("register_draft"):
            return [
                {"status": "ok", "registered": True, "draft_token_count": 2},
                {"status": "skipped", "reason": "no_boundary_proposer"},
            ]
        if method.endswith("draft_status"):
            return [{"status": "ok", "active_requests": 1}]
        return [{"status": "cleared"}, {"status": "skipped"}]


class VLLMCollectiveRPCDraftFeedbackTest(unittest.IsolatedAsyncioTestCase):
    async def test_registers_and_clears_across_workers(self) -> None:
        client = FakeAsyncEngineClient()
        feedback = VLLMCollectiveRPCDraftFeedback(client, timeout=7)
        draft = DraftRequest(
            request_id="main",
            token_ids=(20, 21),
            boundary=DraftBoundary(token_ids=(10, 11)),
            prompt_token_count=4,
        )
        receipt = await feedback.submit(draft)
        await feedback.clear("main")

        method, timeout, args = client.calls[0]
        self.assertEqual(method, "self_speculation_register_draft")
        self.assertEqual(timeout, 7)
        self.assertEqual(args, ("main", [20, 21], [10, 11], 4))
        self.assertEqual(receipt.draft_token_count, 2)
        self.assertEqual(client.calls[1][0], "self_speculation_clear_draft")

    async def test_rejects_invalid_drafts_or_missing_proposers(self) -> None:
        feedback = VLLMCollectiveRPCDraftFeedback(
            FakeAsyncEngineClient(results=[{"status": "skipped"}])
        )
        with self.assertRaisesRegex(ValueError, "token_ids"):
            await feedback.submit(DraftRequest(request_id="x", text="raw"))
        with self.assertRaisesRegex(ValueError, "boundary token_ids"):
            await feedback.submit(
                DraftRequest(request_id="x", token_ids=(1,))
            )
        with self.assertRaisesRegex(VLLMIntegrationError, "no vLLM worker"):
            await feedback.submit(
                DraftRequest(
                    request_id="x",
                    token_ids=(1,),
                    boundary=DraftBoundary(token_ids=(2,)),
                )
            )


class VLLMHTTPRoutesTest(unittest.IsolatedAsyncioTestCase):
    async def test_round_trips_feedback_through_fastapi_routes(self) -> None:
        app = FastAPI()
        engine_client = FakeAsyncEngineClient()
        app.state.engine_client = engine_client
        self.assertTrue(install_vllm_http_routes(app))
        self.assertFalse(install_vllm_http_routes(app))

        client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://vllm",
        )
        feedback = VLLMHTTPDraftFeedback("http://vllm", client=client)
        receipt = await feedback.submit(
            DraftRequest(
                request_id="main/one",
                token_ids=(20, 21),
                boundary=DraftBoundary(text="<tool_call>", token_ids=(10,)),
                prompt_token_count=3,
            )
        )
        status = await feedback.status()
        await feedback.clear("main/one")

        self.assertTrue(receipt.registered)
        self.assertEqual(status["worker_results"][0]["active_requests"], 1)
        self.assertEqual(
            engine_client.calls[-1][2],
            ("main/one",),
        )
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
            "/self-speculation/drafts",
            json={"request_id": "x", "token_ids": [1]},
        )
        self.assertEqual(response.status_code, 422)
        await client.aclose()


if __name__ == "__main__":
    unittest.main()
