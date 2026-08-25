from __future__ import annotations

import json
import unittest

import httpx

from self_speculation import (
    DraftBoundary,
    DraftFeedbackHTTPError,
    DraftRequest,
    HTTPDraftFeedback,
    SporkHTTPDraftFeedback,
    ToolCall,
)


class HTTPDraftFeedbackTest(unittest.IsolatedAsyncioTestCase):
    async def test_sends_portable_registration_and_request_scoped_clear(self) -> None:
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            if request.method == "POST":
                return httpx.Response(
                    200,
                    json={
                        "registered": True,
                        "request_id": "turn/one",
                        "accepted_token_count": 2,
                    },
                )
            return httpx.Response(204)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        feedback = HTTPDraftFeedback(
            "http://engine", client=client, api_key="secret"
        )
        draft = DraftRequest(
            request_id="turn/one",
            text='{"name":"search"}',
            token_ids=(1, 2),
            boundary=DraftBoundary(text="<tool_call>", token_ids=(9,)),
            prompt_token_count=8,
            tool_calls=(ToolCall("search", {"q": "x"}, format="tagged_json"),),
            metadata={"source": "fork"},
        )
        receipt = await feedback.submit(draft)
        await feedback.clear(draft.request_id)

        body = json.loads(requests[0].content)
        self.assertEqual(str(requests[0].url), "http://engine/drafts")
        self.assertEqual(requests[0].headers["authorization"], "Bearer secret")
        self.assertEqual(body["boundary"]["token_ids"], [9])
        self.assertEqual(body["tool_calls"][0]["name"], "search")
        self.assertEqual(receipt.accepted_token_count, 2)
        self.assertEqual(requests[1].method, "DELETE")
        self.assertEqual(str(requests[1].url), "http://engine/drafts/turn%2Fone")
        await client.aclose()

    async def test_matches_original_spork_wire_contract(self) -> None:
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            if request.url.path.endswith("set_tokens"):
                return httpx.Response(
                    200,
                    json={"status": "ok", "request_id": "r", "n_tokens": 3},
                )
            return httpx.Response(200, json={"status": "cleared"})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        feedback = SporkHTTPDraftFeedback("http://vllm", client=client)
        receipt = await feedback.submit(
            DraftRequest(
                request_id="r",
                token_ids=(4, 5, 6),
                prompt_token_count=11,
            )
        )
        await feedback.clear("r")

        self.assertEqual(
            json.loads(requests[0].content),
            {"request_id": "r", "draft_tokens": [4, 5, 6], "prompt_len": 11},
        )
        self.assertEqual(requests[0].url.path, "/spork/set_tokens")
        self.assertEqual(requests[1].method, "POST")
        self.assertEqual(requests[1].url.path, "/spork/clear")
        self.assertEqual(receipt.draft_token_count, 3)
        await client.aclose()

    async def test_returns_request_scoped_verification_on_clear(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "DELETE":
                return httpx.Response(
                    200,
                    json={
                        "status": "cleared",
                        "verification": {
                            "num_spec_steps": 1,
                            "num_draft_tokens": 3,
                            "num_accepted_draft_tokens": 2,
                            "num_rejected_draft_tokens": 1,
                            "steps": [
                                {
                                    "candidate_index": 1,
                                    "candidate_id": "fallback",
                                    "drafted_tokens": 3,
                                    "accepted_tokens": 2,
                                }
                            ],
                        },
                    },
                )
            return httpx.Response(200, json={"registered": True})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        feedback = HTTPDraftFeedback("http://engine", client=client)

        outcome = await feedback.clear("main")

        self.assertIsNotNone(outcome)
        self.assertEqual(outcome.accepted_tokens if outcome else None, 2)
        self.assertEqual(
            outcome.steps[0].candidate_id if outcome else None,
            "fallback",
        )
        await client.aclose()

    async def test_rejects_missing_tokens_and_sidecar_errors(self) -> None:
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200, json={"status": "error", "error": "no drafter"}
                )
            )
        )
        spork = SporkHTTPDraftFeedback("http://vllm", client=client)
        with self.assertRaisesRegex(ValueError, "tokenized"):
            await spork.submit(DraftRequest(request_id="r", text="raw"))
        with self.assertRaisesRegex(DraftFeedbackHTTPError, "no drafter"):
            await spork.submit(DraftRequest(request_id="r", token_ids=(1,)))
        await client.aclose()


if __name__ == "__main__":
    unittest.main()
