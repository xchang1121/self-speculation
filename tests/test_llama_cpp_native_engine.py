from __future__ import annotations

import importlib.util
import unittest

from self_speculation import (
    BoundaryDraftStore,
    DraftBoundary,
    DraftBundle,
    DraftRequest,
    InferenceRequest,
    LlamaCppBoundaryDraftModel,
    LlamaCppPythonEngine,
)


HAS_NUMPY = importlib.util.find_spec("numpy") is not None


class FakeLlama:
    def __init__(self, draft_model=None) -> None:
        self.draft_model = draft_model
        self.calls = []
        self.proposed = ()

    def tokenize(self, text, *, add_bos, special):
        del text, special
        return ([1] if add_bos else []) + [4, 5]

    def create_completion(self, **kwargs):
        self.calls.append(("completion", kwargs))
        if self.draft_model is not None:
            import numpy as np

            self.proposed = tuple(
                int(token)
                for token in self.draft_model(
                    np.asarray([1, 9], dtype=np.intc)
                ).tolist()
            )
        yield {
            "choices": [{"index": 0, "text": "hello ", "finish_reason": None}]
        }
        yield {
            "choices": [{"index": 0, "text": "world", "finish_reason": "stop"}],
            "usage": {"completion_tokens": 2},
        }

    def create_chat_completion(self, **kwargs):
        self.calls.append(("chat", kwargs))
        yield {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call-1",
                                "function": {
                                    "name": "weather",
                                    "arguments": '{"city":',
                                },
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        }


class LlamaCppPythonEngineTest(unittest.IsolatedAsyncioTestCase):
    def test_default_draft_cap_matches_the_evidence_backed_store_cap(self) -> None:
        store = BoundaryDraftStore()
        draft_model = LlamaCppBoundaryDraftModel(store)

        self.assertEqual(draft_model.max_tokens, 28)
        self.assertEqual(draft_model.max_tokens, store.max_draft_tokens)

    @unittest.skipUnless(HAS_NUMPY, "llama-cpp-python requires numpy")
    async def test_injects_request_scoped_candidates_into_draft_model(self) -> None:
        store = BoundaryDraftStore(max_draft_tokens=4)
        draft_model = LlamaCppBoundaryDraftModel(store, max_tokens=4)
        llama = FakeLlama(draft_model)
        engine = LlamaCppPythonEngine(llama, draft_model=draft_model)
        await engine.submit(
            DraftBundle(
                "llama-main",
                (
                    DraftRequest(
                        request_id="llama-main",
                        token_ids=(10, 11),
                        boundary=DraftBoundary(token_ids=(9,)),
                        prompt_token_count=1,
                    ),
                ),
            )
        )

        chunks = [
            chunk
            async for chunk in engine.stream(
                InferenceRequest(
                    prompt="prompt",
                    request_id="llama-main",
                    max_tokens=8,
                )
            )
        ]

        self.assertEqual("".join(chunk.text for chunk in chunks), "hello world")
        self.assertEqual(llama.proposed, (10, 11))
        self.assertEqual(store.snapshot().injections, 1)
        self.assertIsNone(draft_model.active_request_id)
        outcome = await engine.clear("llama-main")
        self.assertIsNotNone(outcome)
        self.assertEqual(outcome.unresolved_proposals if outcome else None, 1)
        self.assertEqual(outcome.unresolved_draft_tokens if outcome else None, 2)
        self.assertEqual(store.snapshot().active_requests, 0)

    async def test_normalizes_native_chat_tool_deltas(self) -> None:
        llama = FakeLlama()
        engine = LlamaCppPythonEngine(llama)
        chunks = [
            chunk
            async for chunk in engine.stream(
                InferenceRequest(
                    messages=({"role": "user", "content": "weather"},),
                    tools=({"type": "function", "function": {"name": "weather"}},),
                )
            )
        ]

        self.assertEqual(chunks[0].tool_call_deltas[0].name, "weather")
        self.assertEqual(chunks[0].tool_call_deltas[0].call_id, "call-1")
        self.assertEqual(llama.calls[0][0], "chat")
        self.assertEqual(llama.calls[0][1]["tools"][0]["type"], "function")

    def test_requires_draft_model_to_be_installed_at_llama_creation(self) -> None:
        draft_model = LlamaCppBoundaryDraftModel(BoundaryDraftStore())
        with self.assertRaisesRegex(ValueError, "construction time|construct"):
            LlamaCppPythonEngine(FakeLlama(), draft_model=draft_model)


if __name__ == "__main__":
    unittest.main()
