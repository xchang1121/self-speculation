from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

import httpx
import numpy as np

from self_speculation import (
    DraftBoundary,
    DraftBundle,
    DraftRequest,
    SGLangHTTPDraftFeedback,
)
from self_speculation.integrations import sglang as integration


class FakeWorker:
    pass


def initialize_worker(worker: FakeWorker) -> None:
    worker.draft_token_num = 4
    worker.enable_overlap = False


class FakeBatch:
    def __init__(self, requests) -> None:
        self.reqs = requests

    def grammar_needs_sync(self) -> bool:
        return False


class SGLangIntegrationTest(unittest.IsolatedAsyncioTestCase):
    async def test_routes_request_scoped_draft_into_ngram_verify_candidates(self) -> None:
        worker = FakeWorker()
        integration._worker_init(initialize_worker, worker)
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            body = json.loads(request.content)
            if request.url.path == "/add_external_corpus":
                count = integration._add_external_corpus(
                    lambda current, corpus_id, chunks: 0,
                    worker,
                    body["corpus_id"],
                    [[99]],
                )
                integration._commit_corpus_load(
                    lambda current, corpus_id, loaded: None,
                    worker,
                    body["corpus_id"],
                    count,
                )
                return httpx.Response(
                    200,
                    json={
                        "success": True,
                        "corpus_id": body["corpus_id"],
                        "loaded_token_count": count,
                    },
                )
            integration._remove_external_corpus(
                lambda current, corpus_id: None,
                worker,
                body["corpus_id"],
            )
            return httpx.Response(200, json={"success": True})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        feedback = SGLangHTTPDraftFeedback("http://sglang", client=client)
        receipt = await feedback.submit(
            DraftBundle(
                request_id="main-request",
                drafts=(
                    DraftRequest(
                        request_id="main-request",
                        token_ids=(10, 11),
                        boundary=DraftBoundary(token_ids=(9,)),
                        metadata={
                            "candidate_id": "fork",
                            "sources": ["actor-fork"],
                        },
                    ),
                    DraftRequest(
                        request_id="main-request",
                        token_ids=(10, 12),
                        boundary=DraftBoundary(token_ids=(9,)),
                        metadata={"candidate_id": "drafter", "sources": ["drafter"]},
                    ),
                ),
            )
        )
        self.assertEqual(
            getattr(worker, integration._STORE_ATTRIBUTE).snapshot().active_requests,
            0,
        )

        original_drafts = np.asarray([2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int64)
        original_masks = np.stack(
            [np.eye(4, dtype=np.int64), np.eye(4, dtype=np.int64)]
        ).reshape(-1)
        batch = FakeBatch(
            [
                SimpleNamespace(
                    rid="main-request", origin_input_ids=[1], output_ids=[9]
                ),
                SimpleNamespace(
                    rid="other-request", origin_input_ids=[1], output_ids=[9]
                ),
            ]
        )
        drafts, masks = integration._prepare_draft_tokens(
            lambda current, current_batch: (original_drafts, original_masks),
            worker,
            batch,
        )

        self.assertTrue(receipt.registered)
        self.assertEqual(receipt.draft_token_count, 2)
        self.assertEqual(drafts.reshape(2, 4)[0].tolist(), [10, 11, 11, 11])
        self.assertEqual(drafts.reshape(2, 4)[1].tolist(), [6, 7, 8, 9])
        self.assertEqual(
            masks.reshape(2, 4, 4)[0].tolist(),
            np.tril(np.ones((4, 4), dtype=np.int64)).tolist(),
        )
        self.assertEqual(
            masks.reshape(2, 4, 4)[1].tolist(),
            np.eye(4, dtype=np.int64).tolist(),
        )
        self.assertEqual(
            getattr(worker, integration._STORE_ATTRIBUTE).snapshot().injections,
            1,
        )
        self.assertEqual(
            getattr(worker, integration._STORE_ATTRIBUTE)
            .snapshot()
            .registered_candidates,
            2,
        )
        self.assertEqual(getattr(worker, integration._PENDING_ATTRIBUTE), {})
        submit_body = json.loads(requests[0].content)
        self.assertEqual(submit_body["documents"], ["_"])
        control = integration._decode_control(submit_body["corpus_id"])
        self.assertIsNotNone(control)
        self.assertEqual(
            [item["metadata"]["candidate_id"] for item in control["drafts"]],
            ["fork", "drafter"],
        )

        await feedback.clear("main-request")
        self.assertEqual(
            getattr(worker, integration._STORE_ATTRIBUTE).snapshot().active_requests,
            0,
        )
        await client.aclose()

    async def test_rejects_incomplete_feedback_payloads(self) -> None:
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(200))
        )
        feedback = SGLangHTTPDraftFeedback("http://sglang", client=client)
        with self.assertRaisesRegex(ValueError, "boundary token_ids"):
            await feedback.submit(
                DraftBundle(
                    "r",
                    (
                        DraftRequest(
                            request_id="r",
                            token_ids=(1,),
                            boundary=DraftBoundary(text="<tool>"),
                            prompt_token_count=2,
                        ),
                    ),
                )
            )
        await client.aclose()

    def test_preserves_normal_corpus_operations_and_clears_store(self) -> None:
        worker = FakeWorker()
        integration._worker_init(initialize_worker, worker)
        calls = []

        result = integration._add_external_corpus(
            lambda current, corpus_id, chunks: calls.append(
                ("add", corpus_id, chunks)
            )
            or 7,
            worker,
            "ordinary-corpus",
            [[1, 2]],
        )
        integration._commit_corpus_load(
            lambda current, corpus_id, count: calls.append(
                ("commit", corpus_id, count)
            ),
            worker,
            "ordinary-corpus",
            result,
        )
        integration._remove_external_corpus(
            lambda current, corpus_id: calls.append(("remove", corpus_id)),
            worker,
            "ordinary-corpus",
        )
        integration._clear_cache_pool(
            lambda current: calls.append(("clear",)), worker
        )

        self.assertEqual(result, 7)
        self.assertEqual(
            calls,
            [
                ("add", "ordinary-corpus", [[1, 2]]),
                ("commit", "ordinary-corpus", 7),
                ("remove", "ordinary-corpus"),
                ("clear",),
            ],
        )

    def test_registers_only_official_around_hooks(self) -> None:
        calls = []
        registry = SimpleNamespace(
            register=lambda target, hook, hook_type: calls.append(
                (target, hook, hook_type)
            )
        )
        around = object()

        integration._register_hooks(registry, around)

        self.assertEqual(len(calls), 7)
        self.assertTrue(
            all(target.startswith("sglang.srt.") for target, _, _ in calls)
        )
        self.assertTrue(all(hook_type is around for _, _, hook_type in calls))

    def test_control_registration_does_not_occupy_corpus_load_slot(self) -> None:
        worker = FakeWorker()
        integration._worker_init(initialize_worker, worker)
        control_id = integration._encode_control(
            {
                "op": "replace",
                "request_id": "r",
                "drafts": [
                    {
                        "token_ids": [10],
                        "boundary_token_ids": [9],
                        "prompt_token_count": 1,
                    }
                ],
            }
        )
        manager = SimpleNamespace(
            _worker=SimpleNamespace(
                add_external_corpus=lambda corpus_id, chunks: integration._add_external_corpus(
                    lambda current, normal_id, normal_chunks: 0,
                    worker,
                    corpus_id,
                    chunks,
                ),
                commit_corpus_load=lambda corpus_id, count: integration._commit_corpus_load(
                    lambda current, normal_id, loaded: None,
                    worker,
                    corpus_id,
                    count,
                ),
            )
        )
        output = integration._external_corpus_manager_add(
            lambda current, request: self.fail("normal loader must not run"),
            manager,
            SimpleNamespace(corpus_id=control_id, token_chunks=[[99]]),
            _output_type=lambda **values: SimpleNamespace(**values),
        )

        self.assertTrue(output.success)
        self.assertEqual(output.loaded_token_count, 1)
        self.assertFalse(hasattr(manager, "_pending_load"))


if __name__ == "__main__":
    unittest.main()
