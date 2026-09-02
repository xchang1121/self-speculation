from __future__ import annotations

import json
import unittest

import httpx

from self_speculation import (
    InferenceRequest,
    LlamaCppEngine,
    OpenAICompatibleEngine,
    OpenAIStreamError,
    SGLangEngine,
    TGIEngine,
    VLLMEngine,
    fit_request_to_context,
)


def sse(*payloads: object) -> bytes:
    lines = [f"data: {json.dumps(payload)}\n\n" for payload in payloads]
    lines.append("data: [DONE]\n\n")
    return "".join(lines).encode()


class OpenAICompatibleEngineTest(unittest.IsolatedAsyncioTestCase):
    async def test_exposes_injected_exact_prompt_accounting(self) -> None:
        client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(500)))
        engine = OpenAICompatibleEngine(
            "http://engine/v1",
            client=client,
            max_context_tokens=10,
            prompt_token_counter=lambda request: len(request.prompt or ""),
        )

        bounded, budget = await fit_request_to_context(
            engine, InferenceRequest(prompt="12345678", max_tokens=9)
        )

        self.assertEqual(bounded.max_tokens, 2)
        self.assertEqual(budget.prompt_tokens if budget else None, 8)
        await client.aclose()

    async def test_streams_raw_completion_with_logprobs_and_tokens(self) -> None:
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=sse(
                    {
                        "choices": [
                            {
                                "index": 0,
                                "text": "tool",
                                "token_ids": [7],
                                "logprobs": {
                                    "tokens": ["tool"],
                                    "token_logprobs": [-0.1],
                                    "top_logprobs": [{"tool": -0.1}],
                                },
                                "finish_reason": None,
                            }
                        ]
                    },
                    {"choices": [{"index": 0, "text": "", "finish_reason": "stop"}]},
                ),
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        engine = OpenAICompatibleEngine(
            "http://engine/v1", client=client, api_key="secret"
        )
        chunks = [
            chunk
            async for chunk in engine.stream(
                InferenceRequest(
                    prompt="PROMPT",
                    model="model",
                    max_tokens=10,
                    stop=("END",),
                    extra={"logprobs": 5, "stream": False},
                )
            )
        ]

        body = json.loads(requests[0].content)
        self.assertEqual(str(requests[0].url), "http://engine/v1/completions")
        self.assertEqual(requests[0].headers["authorization"], "Bearer secret")
        self.assertTrue(body["stream"])
        self.assertEqual(body["prompt"], "PROMPT")
        self.assertEqual(body["stop"], ["END"])
        self.assertEqual(chunks[0].token_ids, (7,))
        self.assertEqual(chunks[0].logprobs[0].top_logprobs, {"tool": -0.1})
        self.assertEqual(chunks[1].finish_reason, "stop")
        await client.aclose()

    async def test_streams_chat_reasoning_and_parallel_tool_deltas(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            self.assertEqual(body["messages"][0]["content"], "question")
            self.assertEqual(body["tools"][0]["type"], "function")
            return httpx.Response(
                200,
                content=sse(
                    {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"reasoning_content": "thinking"},
                                "finish_reason": None,
                            }
                        ]
                    },
                    {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": "a",
                                            "function": {
                                                "name": "first",
                                                "arguments": '{"x":',
                                            },
                                        },
                                        {
                                            "index": 1,
                                            "id": "b",
                                            "function": {
                                                "name": "second",
                                                "arguments": "{}",
                                            },
                                        },
                                    ]
                                },
                                "finish_reason": None,
                            }
                        ]
                    },
                    {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [
                                        {"index": 0, "function": {"arguments": "1}"}}
                                    ]
                                },
                                "finish_reason": "tool_calls",
                            }
                        ],
                        "usage": {"completion_tokens": 4},
                    },
                ),
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        engine = OpenAICompatibleEngine("http://engine/v1", client=client)
        chunks = [
            chunk
            async for chunk in engine.stream(
                InferenceRequest(
                    messages=({"role": "user", "content": "question"},),
                    tools=({"type": "function", "function": {"name": "first"}},),
                )
            )
        ]

        self.assertEqual(chunks[0].reasoning, "thinking")
        self.assertEqual([item.name for item in chunks[1].tool_call_deltas], ["first", "second"])
        self.assertEqual(chunks[2].tool_call_deltas[0].arguments, "1}")
        self.assertEqual(chunks[2].usage, {"completion_tokens": 4})
        await client.aclose()

    async def test_surfaces_sse_error_objects(self) -> None:
        transport = httpx.MockTransport(
            lambda request: httpx.Response(200, content=sse({"error": {"message": "bad"}}))
        )
        client = httpx.AsyncClient(transport=transport)
        engine = OpenAICompatibleEngine("http://engine/v1", client=client)
        with self.assertRaises(OpenAIStreamError):
            _ = [
                chunk
                async for chunk in engine.stream(InferenceRequest(prompt="x"))
            ]
        await client.aclose()

    async def test_named_compatible_adapters_share_the_contract(self) -> None:
        client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200)))
        engines = [
            VLLMEngine("http://x/v1", client=client),
            SGLangEngine("http://x/v1", client=client),
            TGIEngine("http://x/v1", client=client),
            LlamaCppEngine("http://x/v1", client=client),
        ]
        self.assertEqual(
            [engine.name for engine in engines],
            ["vllm", "sglang", "tgi", "llama.cpp"],
        )
        _, vllm_body = engines[0]._payload(
            InferenceRequest(prompt="x", request_id="stable-id")
        )
        _, sglang_body = engines[1]._payload(
            InferenceRequest(prompt="x", request_id="stable-id")
        )
        self.assertEqual(vllm_body["request_id"], "stable-id")
        self.assertEqual(sglang_body["rid"], "stable-id")
        self.assertEqual(vllm_body["stream_options"], {"include_usage": True})
        self.assertEqual(sglang_body["stream_options"], {"include_usage": True})
        await client.aclose()

    async def test_normalizes_server_cache_accounting(self) -> None:
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    content=sse(
                        {
                            "choices": [],
                            "usage": {
                                "prompt_tokens": 20,
                                "prompt_tokens_details": {"cached_tokens": 16},
                            },
                        }
                    ),
                )
            )
        )
        engine = VLLMEngine(
            "http://engine/v1",
            client=client,
            prefix_cache=True,
        )

        chunks = [
            chunk
            async for chunk in engine.stream(InferenceRequest(prompt="prompt"))
        ]

        self.assertEqual(chunks[0].usage["cache_read_tokens"], 16)
        await client.aclose()


if __name__ == "__main__":
    unittest.main()
